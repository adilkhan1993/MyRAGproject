from slowapi import Limiter
from slowapi.util import get_remote_address

# Создаем лимитер, который будет различать пользователей по их IP-адресам
limiter = Limiter(key_func=get_remote_address)