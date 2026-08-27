import ipaddress
import socket
from urllib.parse import urlparse


# Public internet ranges only
ALLOWED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/0"), # Any public IPv4
]

ALWAYS_BLOCKED = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10")
]

ALLOWED_PROTOCOLS = {"https", "http"}


class SSRFError(Exception):
    pass


def validate_url(url: str) -> str:
    """
    Raises SSRFError if the URL resolves to a disallowed target,
    returns the resolved IP for logging.
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_PROTOCOLS:
        raise SSRFError(f"Disallowed scheme: {parsed_url.scheme}")

    if not parsed_url.hostname:
        raise SSRFError("No hostname in URL")

    hostname = parsed_url.hostname

    # Resolve DNS (looks up IP address)
    try:
        addrinfo = socket.getaddrinfo(hostname, None) # family, type, proto, canonname, sockaddr
    except socket.gaierror as e:
        raise SSRFError(f"DNS resolution failed: {e}")

    for family, _, _, _, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])

        # Block IP addresses
        for net in ALWAYS_BLOCKED:
            if ip in net:
                raise SSRFError(f"Resolved to always-blocked range: {ip} in {net}")

        # IP must match allowed network
        if not any(ip in net for net in ALLOWED_NETWORKS):
            raise SSRFError(f"Resolved IP not in allowed ranges: {ip}")

    return str(addrinfo[0][4][0])
