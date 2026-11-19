import http.client
import ipaddress
import json
import queue
import socket
import ssl
import time
import threading
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin, urlencode

MAX_AGE_DAYS = 2
VERIFY_TIMEOUT = 8
MAX_PAGE_BYTES = 1_000_000


def parse_publication_date(value):
    if not isinstance(value, str) or 'T' not in value:
        raise ValueError('Publication timestamp requires a time and timezone')
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Publication timezone missing')
    return parsed.astimezone(timezone.utc)


def is_current_article(article, reference_time):
    if article.get('freshness_status') != 'VERIFIED_FRESH':
        return False
    try:
        published = parse_publication_date(article.get('original_published_at_utc'))
        return timedelta(0) <= reference_time - published <= timedelta(days=MAX_AGE_DAYS)
    except (ValueError, TypeError):
        return False


def require_fresh_event_sources(events):
    # Fail closed for cached/legacy events and mixed-source summaries or claims.
    for event in events:
        sources = event.get('sources', [])
        if not sources or not all(
            is_current_article(
                source, parse_publication_date(source.get('freshness_checked_at_utc')))
            for source in sources
        ):
            raise ValueError('Current-market event contains ineligible publication evidence')


def resolve_public_addresses(hostname, port, deadline):
    # System DNS has no socket timeout. Bound the caller with a daemon worker.
    outcome = queue.Queue(maxsize=1)

    def resolve():
        try:
            outcome.put(socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            outcome.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses = outcome.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty as exc:
        raise TimeoutError('Publisher DNS verification timed out') from exc
    if isinstance(addresses, Exception):
        raise addresses
    if not addresses:
        raise ValueError('Publisher has no addresses')
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError('Non-public publisher destination')
    return addresses


def interrupt_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


def fetch_public_page(url, deadline, *, body=None):
    """Validate every hop and pin the connection to a checked public IP."""
    for _ in range(5):
        parts = urlsplit(url)
        if (parts.scheme not in ('http', 'https') or not parts.hostname
                or parts.username or parts.password
                or parts.port not in (None, 80, 443)):
            raise ValueError('Unsafe publisher URL')
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        addresses = resolve_public_addresses(parts.hostname, port, deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Publication verification deadline exceeded')
        connection = http.client.HTTPConnection(parts.hostname, port, timeout=remaining)
        sock = None
        watchdog = None
        try:
            sock = socket.create_connection((addresses[0][4][0], port), timeout=remaining)
            if parts.scheme == 'https':
                sock = ssl.create_default_context().wrap_socket(
                    sock, server_hostname=parts.hostname, do_handshake_on_connect=False)
            connection.sock = sock
            watchdog = threading.Timer(max(0, deadline - time.monotonic()), interrupt_socket, args=(sock,))
            watchdog.daemon = True
            watchdog.start()
            if parts.scheme == 'https':
                sock.do_handshake()
            path = parts.path or '/'
            if parts.query:
                path += '?' + parts.query
            headers = {
                'User-Agent': 'NiftyResearchPrototype/0.1',
                'Accept-Encoding': 'identity',
            }
            if body is not None:
                headers['Content-Type'] = 'application/x-www-form-urlencoded;charset=UTF-8'
            connection.request('POST' if body is not None else 'GET', path,
                               body=body, headers=headers)
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                if body is not None:
                    raise ValueError('Google News resolution POST redirected')
                location = response.getheader('Location')
                if not location:
                    raise ValueError('Redirect missing destination')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f'Publisher HTTP {response.status}')
            chunks = []
            size = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Publication verification deadline exceeded')
                sock.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_PAGE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    raise ValueError('Publisher page exceeds size limit')
            return url, b''.join(chunks).decode('utf-8', errors='replace')
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
            if sock is not None:
                sock.close()
    raise ValueError('Too many publisher redirects')


def is_google_news_url(url):
    return (urlsplit(url).hostname or '').rstrip('.').lower() in (
        'news.google.com', 'news.google.co.in')


class GoogleArticleMetadata(HTMLParser):
    """Read explicit article identity fields, never arbitrary navigation links."""

    def __init__(self, article_id):
        super().__init__()
        self.article_id = article_id
        self.publisher_urls = set()
        self.rpc_parameters = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'link' and 'canonical' in (attrs.get('rel') or '').lower().split():
            self.publisher_urls.add(attrs.get('href', ''))
        if tag == 'meta' and (attrs.get('property') or '').lower() == 'og:url':
            self.publisher_urls.add(attrs.get('content', ''))
        if attrs.get('data-n-a-id') == self.article_id:
            if attrs.get('data-n-au'):
                self.publisher_urls.add(attrs['data-n-au'])
            if tag == 'a' and attrs.get('href'):
                self.publisher_urls.add(attrs['href'])
            signature = attrs.get('data-n-a-sg')
            timestamp = attrs.get('data-n-a-ts') or ''
            if signature and timestamp.isascii() and timestamp.isdigit():
                self.rpc_parameters.add((int(timestamp), signature))


def decode_google_resolution_response(text):
    """Read typed RPC frames, including length-prefixed batchexecute responses."""
    urls = set()
    for line in text.splitlines():
        if not line.lstrip().startswith('['):
            continue
        try:
            frames = json.loads(line)
        except ValueError:
            continue
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if (not isinstance(frame, list) or len(frame) < 3
                    or frame[:2] != ['wrb.fr', 'Fbv4je']
                    or not isinstance(frame[2], str)):
                continue
            try:
                payload = json.loads(frame[2])
            except ValueError:
                continue
            if (isinstance(payload, list) and len(payload) >= 2
                    and payload[0] == 'garturlres' and isinstance(payload[1], str)):
                urls.add(payload[1])
    if len(urls) != 1:
        raise ValueError('Google News resolution returned missing or ambiguous publisher URL')
    return urls.pop()


def resolve_publisher_page(url, deadline):
    resolved, html = fetch_public_page(url, deadline)
    if not is_google_news_url(resolved):
        return resolved, html

    path = urlsplit(resolved).path.strip('/').split('/')
    if len(path) < 2 or path[-2] not in ('articles', 'read'):
        raise ValueError('Google News link did not resolve to an article page')
    article_id = path[-1]
    metadata = GoogleArticleMetadata(article_id)
    metadata.feed(html)
    metadata.close()
    publisher_urls = {
        urljoin(resolved, candidate) for candidate in metadata.publisher_urls
        if candidate and not is_google_news_url(urljoin(resolved, candidate))
    }
    if len(publisher_urls) > 1:
        raise ValueError('Ambiguous publisher URLs in Google News article metadata')
    if publisher_urls:
        publisher_url = publisher_urls.pop()
    else:
        if len(metadata.rpc_parameters) != 1:
            raise ValueError('Google News link did not resolve to a publisher')
        timestamp, signature = metadata.rpc_parameters.pop()
        # Google uses an article-bound signed RPC for opaque article IDs.
        context = [
            ['X', 'X', ['X', 'X'], None, None, 1, 1, 'US:en', None,
             1, None, None, None, None, None, 0, 1],
            'X', 'X', 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
        ]
        request = ['garturlreq', context, article_id, timestamp, signature]
        rpc = [[['Fbv4je', json.dumps(request), None, 'generic']]]
        body = urlencode({'f.req': json.dumps(rpc)}).encode('utf-8')
        endpoint = 'https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je'
        rpc_url, response = fetch_public_page(endpoint, deadline, body=body)
        if not is_google_news_url(rpc_url):
            raise ValueError('Unexpected Google News resolution endpoint')
        publisher_url = decode_google_resolution_response(response)
    if is_google_news_url(publisher_url):
        raise ValueError('Google News resolution returned another Google News URL')
    # Discovered URLs and all subsequent redirects use the same pinned-IP transport.
    resolved, html = fetch_public_page(publisher_url, deadline)
    if is_google_news_url(resolved):
        raise ValueError('Publisher destination redirected back to Google News')
    return resolved, html


class PublicationMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.dates = []
        self.in_json = False
        self.script = []
        self.invalid = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and ((attrs.get('property') or '').lower() == 'article:published_time'
                              or (attrs.get('itemprop') or '').lower() == 'datepublished'):
            self.dates.append(attrs.get('content'))
        if tag == 'script' and (attrs.get('type') or '').lower() == 'application/ld+json':
            self.in_json = True
            self.script = []

    def handle_data(self, data):
        if self.in_json:
            self.script.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.in_json:
            self.in_json = False
            try:
                structured = json.loads(''.join(self.script))
            except ValueError:
                # Broken unrelated JSON-LD supplies no publication evidence.
                return
            try:
                self.read_structured(structured)
            except (TypeError, RecursionError):
                self.invalid = True

    def read_structured(self, value):
        if isinstance(value, list):
            for item in value:
                self.read_structured(item)
        elif isinstance(value, dict):
            types = value.get('@type', [])
            if isinstance(types, str):
                types = [types]
            if not isinstance(types, list):
                types = []
            if any(t in ('Article', 'NewsArticle', 'ReportageNewsArticle', 'BlogPosting') for t in types):
                if 'datePublished' in value:
                    self.dates.append(value['datePublished'])
            if '@graph' in value:
                self.read_structured(value['@graph'])


def verify_original_publication(url, reference_time):
    result = {
        'original_article_url': None,
        'original_published_at_utc': None,
        'freshness_status': 'UNVERIFIED_DATE',
        'freshness_reason': '',
        'publication_verification_failed': False,
        'freshness_checked_at_utc': reference_time.isoformat(),
    }
    try:
        resolved, html = resolve_publisher_page(url, time.monotonic() + VERIFY_TIMEOUT)
        result['original_article_url'] = resolved
        metadata = PublicationMetadata()
        metadata.feed(html)
        metadata.close()
        if metadata.invalid or metadata.in_json or not metadata.dates:
            raise ValueError('Missing or malformed original publication metadata')
        dates = {parse_publication_date(value) for value in metadata.dates}
        if len(dates) != 1:
            raise ValueError('Conflicting original publication metadata')
        published = dates.pop()
        result['original_published_at_utc'] = published.isoformat()
        age = reference_time - published
        if age < timedelta(0):
            result['freshness_reason'] = 'Original publication date is in the future'
            result['publication_verification_failed'] = True
        elif age > timedelta(days=MAX_AGE_DAYS):
            result.update(freshness_status='STALE', freshness_reason='Original publication exceeds seven-day window')
        else:
            result.update(freshness_status='VERIFIED_FRESH', freshness_reason='Publisher original publication is within seven-day window')
    except (OSError, ValueError, TypeError, http.client.HTTPException, RecursionError) as exc:
        result['freshness_reason'] = str(exc)
        result['publication_verification_failed'] = True
    return result
