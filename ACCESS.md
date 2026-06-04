# Доступ к квантованной модели (демо, без VPN)

Квантованный **Qwen3-8B W8A8** (SmoothQuant+GPTQ, llm-compressor) поднят в vLLM
на vv_h200 и выведен наружу публичным туннелем. OpenAI-совместимый API.

- **Base URL**: `https://28d81a7e5fc51a.lhr.life/v1`
- **Модель**: `qwen3-8b-w8a8`
- **API-ключ**: взять из `vv_h200:~/a2_pro_kvant/.api_key` (или у Андрея)

## Проверка (curl)
```bash
curl -H "Authorization: Bearer <KEY>" https://28d81a7e5fc51a.lhr.life/v1/models

curl -H "Authorization: Bearer <KEY>" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-w8a8","messages":[{"role":"user","content":"Привет!"}],"max_tokens":100}' \
  https://28d81a7e5fc51a.lhr.life/v1/chat/completions
```

## Python (openai)
```python
from openai import OpenAI
client = OpenAI(base_url="https://28d81a7e5fc51a.lhr.life/v1", api_key="<KEY>")
r = client.chat.completions.create(
    model="qwen3-8b-w8a8",
    messages=[{"role": "user", "content": "Привет!"}],
    max_tokens=100,
)
print(r.choices[0].message.content)
```

## Важно
- URL туннеля **временный** (localhost.run): при перезапуске туннеля он сменится.
  Постоянный вариант — свой домен через cloudflared named tunnel или проброс порта.
- Внутри сети МФТИ (VPN) можно ходить напрямую: `http://10.0.116.11:8011/v1`.
- Qwen3 по умолчанию «думает» (`<think>`): для коротких ответов добавляйте
  `/no_think` в конец промпта или `"chat_template_kwargs": {"enable_thinking": false}`.

## Что где живёт (vv_h200)
- vLLM сервер: tmux `kv_serve_w8a8` (GPU6, порт 8011), лог `/tmp/kv_serve_w8a8.log`
- Туннель: tmux `kv_tunnel2` (`ssh -R ... localhost.run`), лог `/tmp/kv_tunnel2.log`
- Чекпоинт: `~/a2_pro_kvant/checkpoints/Qwen3-8B-W8A8` (8.8 ГБ, compressed-tensors)
- Перезапуск сервера: см. README, раздел «Запуск»
