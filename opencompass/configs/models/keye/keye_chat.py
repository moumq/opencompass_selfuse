from opencompass.models import KeyeChat

# Configure ckpts here; each ckpt becomes one model entry automatically.
KEYE_CKPTS = [
    'istep0000200_2nd_autothink',
]

COMMON_CFG = dict(
    type=KeyeChat,
    key='EMPTY',  # local service without auth can use placeholder
    api_base='http://127.0.0.1:15553/v1',
    temperature=0,
    max_tokens=10240,
    max_seq_len=32768,
    retry=60,
    timeout=180,
    verbose=True,
    query_per_second=1,
    batch_size=1,
    img_detail='high',
)

models = [
    dict(
        abbr=f'keye-{ckpt.replace("_", "-")}',
        ckpt=ckpt,
        **COMMON_CFG,
    ) for ckpt in KEYE_CKPTS
]
