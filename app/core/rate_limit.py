from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings


# Applied per client IP to every API route by SlowAPIASGIMiddleware; static files are not limited.
limiter = Limiter(key_func=get_remote_address, default_limits=[get_settings().rate_limit])
