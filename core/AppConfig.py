import os


GITHUB_TOKEN_ENV = "LIFECALOR_GITHUB_TOKEN"
EM_FREQUENCY_RESULT_TYPES = {"ROI_stft", "ROI_cwt"}


def get_github_auth_header() -> str:
    token = os.environ.get(GITHUB_TOKEN_ENV, "").strip()
    if not token:
        return ""
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def build_github_headers(extra_headers=None) -> dict:
    headers = {
        "User-Agent": "Carrier-Lifetime-Calculator",
    }
    if extra_headers:
        headers.update(extra_headers)
    auth_header = get_github_auth_header()
    if auth_header:
        headers["Authorization"] = auth_header
    return headers


def is_em_frequency_result(processed_type) -> bool:
    return processed_type in EM_FREQUENCY_RESULT_TYPES
