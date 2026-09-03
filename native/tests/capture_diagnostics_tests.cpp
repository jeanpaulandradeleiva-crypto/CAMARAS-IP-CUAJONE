// SPDX-License-Identifier: AGPL-3.0-only

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#endif

#include "cuajone/capture.hpp"
#include "cuajone/cli.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace cuajone;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void testAuthorityParsingAndSafeFormatting() {
    const auto authority = parseRtspAuthority(
        "rtsps://account:token@[2001:db8::1]:8554/live/channel?profile=primary");
    require(authority.host == "2001:db8::1" && authority.port == 8554,
        "RTSPS authority parser did not preserve only the host and port");
    const auto default_port = parseRtspAuthority("rtsp://camera.example/live");
    require(default_port.host == "camera.example" && default_port.port == 554,
        "RTSP authority parser did not apply the default port");

    const std::string message = formatRtspOpenFailure(
        authority.host, authority.port, RtspReachabilityReason::TcpTimeout);
    require(message.find("[reason=tcp_timeout]") != std::string::npos
            && message.find("endpoint=[2001:db8::1]:8554") != std::string::npos,
        "RTSP timeout diagnostic did not include the stable category and safe endpoint");
    for (const std::string forbidden : {"account", "token", "/live", "profile=primary", "rtsps://"}) {
        require(message.find(forbidden) == std::string::npos,
            "RTSP diagnostic exposed a credential-bearing URL component: " + forbidden);
    }
}

void testAddressPathMapping() {
    const auto ipv4 = parseRtspAuthority("rtsp://172.19.90.72:554/live");
    require(classifyRtspAddressPath(ipv4.host) == RtspAddressPath::DirectAddress,
        "Literal IPv4 did not select the direct-address preflight path");

    const auto ipv6 = parseRtspAuthority("rtsp://[2001:db8::1]:8554/live");
    require(classifyRtspAddressPath(ipv6.host) == RtspAddressPath::DirectAddress,
        "Bracketed literal IPv6 did not select the direct-address preflight path");

    require(classifyRtspAddressPath("camera.example") == RtspAddressPath::HostnameResolution,
        "Hostname did not select the hostname-resolution preflight path");

#ifdef _WIN32
    constexpr int hostname_resolution_failure = WSAHOST_NOT_FOUND;
#else
    constexpr int hostname_resolution_failure = 1;
#endif
    require(classifyRtspResolutionFailure(
                classifyRtspAddressPath(ipv4.host), hostname_resolution_failure)
            == RtspReachabilityReason::Unknown,
        "Literal IPv4 failure was incorrectly classified as dns_failure");
    require(classifyRtspResolutionFailure(
                classifyRtspAddressPath(ipv6.host), hostname_resolution_failure)
            == RtspReachabilityReason::Unknown,
        "Literal IPv6 failure was incorrectly classified as dns_failure");
}

void testConnectErrorClassification() {
#ifdef _WIN32
    require(classifyRtspConnectError(WSAENETUNREACH) == RtspReachabilityReason::NoRoute,
        "Network-unreachable error was not classified as no_route");
    require(classifyRtspConnectError(WSAETIMEDOUT) == RtspReachabilityReason::TcpTimeout,
        "Timed-out connection was not classified as tcp_timeout");
    require(classifyRtspConnectError(WSAECONNREFUSED) == RtspReachabilityReason::ConnectionRefused,
        "Refused connection was not classified as connection_refused");
    require(classifyRtspConnectError(WSAEINVAL) == RtspReachabilityReason::Unknown,
        "Unrecognized socket error was not classified as unknown");
    require(classifyRtspResolutionFailure(
                RtspAddressPath::HostnameResolution, WSAHOST_NOT_FOUND)
            == RtspReachabilityReason::DnsFailure,
        "Hostname-resolution failure was not classified as dns_failure");
    require(classifyRtspResolutionFailure(
                RtspAddressPath::HostnameResolution, WSAEINVAL)
            == RtspReachabilityReason::Unknown,
        "Unrecognized resolver/API error was not classified as unknown");
#else
    require(classifyRtspConnectError(0) == RtspReachabilityReason::Unknown,
        "Unsupported socket errors must retain the safe unknown fallback");
    require(classifyRtspResolutionFailure(
                RtspAddressPath::HostnameResolution, 0)
            == RtspReachabilityReason::Unknown,
        "Unsupported resolver errors must retain the safe unknown fallback");
#endif
}

void testPreflightDecisionMapping() {
    for (const RtspReachabilityReason reason : {
             RtspReachabilityReason::DnsFailure,
             RtspReachabilityReason::NoRoute,
             RtspReachabilityReason::TcpTimeout,
             RtspReachabilityReason::ConnectionRefused,
         }) {
        require(shouldSkipRtspOpen(reason),
            "Preflight failure did not skip the RTSP open: " + std::to_string(static_cast<int>(reason)));
    }
    for (const RtspReachabilityReason reason : {
             RtspReachabilityReason::RtspHandshakeFailed,
             RtspReachabilityReason::Unknown,
         }) {
        require(!shouldSkipRtspOpen(reason),
            "Non-blocking preflight result skipped the RTSP open: "
                + std::to_string(static_cast<int>(reason)));
    }
}

void testStableReasonMessages() {
    for (const auto& [reason, category] : {
             std::pair{RtspReachabilityReason::DnsFailure, "dns_failure"},
             std::pair{RtspReachabilityReason::NoRoute, "no_route"},
             std::pair{RtspReachabilityReason::TcpTimeout, "tcp_timeout"},
             std::pair{RtspReachabilityReason::ConnectionRefused, "connection_refused"},
             std::pair{RtspReachabilityReason::RtspHandshakeFailed, "rtsp_handshake_failed"},
             std::pair{RtspReachabilityReason::Unknown, "unknown"},
         }) {
        require(formatRtspOpenFailure("camera.example", 554, reason).find(
                    "[reason=" + std::string(category) + "]") != std::string::npos,
            "RTSP diagnostic did not preserve the stable reason category: " + std::string(category));
    }

    const std::string no_route = formatRtspOpenFailure(
        "192.0.2.20", 554, RtspReachabilityReason::NoRoute);
    require(no_route.find("[reason=no_route]") != std::string::npos
            && no_route.find("may be affected by routing, VLAN, firewall, or host availability")
                != std::string::npos,
        "No-route message did not preserve the qualified network-path interpretation");

    const std::string handshake = formatRtspOpenFailure(
        "camera.example", 554, RtspReachabilityReason::RtspHandshakeFailed);
    require(handshake.find("[reason=rtsp_handshake_failed]") != std::string::npos
            && handshake.find("TCP connectivity succeeded") != std::string::npos,
        "RTSP handshake message did not require established TCP connectivity");

    const std::string unknown = formatRtspOpenFailure(
        "", 0, RtspReachabilityReason::Unknown);
    require(unknown.find("[reason=unknown]") != std::string::npos
            && unknown.find("endpoint=unavailable") != std::string::npos
            && unknown.find("OpenCV could not open the source") != std::string::npos,
        "Unknown diagnostic did not retain the generic OpenCV fallback");

    const std::string progress = formatRtspOpenProgress("2001:db8::1", 8554);
    require(progress == "RTSP open [endpoint=[2001:db8::1]:8554]: checking reachability before opening RTSP.",
        "RTSP progress message did not preserve its stable safe format");
    for (const std::string forbidden : {"account", "token", "/live", "profile=primary", "rtsps://"}) {
        require(progress.find(forbidden) == std::string::npos,
            "RTSP progress message exposed a credential-bearing URL component: " + forbidden);
    }
}

}  // namespace

int main() {
    try {
        testAuthorityParsingAndSafeFormatting();
        testAddressPathMapping();
        testConnectErrorClassification();
        testPreflightDecisionMapping();
        testStableReasonMessages();
        std::cout << "PASS: capture diagnostics\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: capture diagnostics: " << error.what() << '\n';
        return 1;
    }
}
