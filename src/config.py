MODEL_CONFIGS = {
    'e5-small': {
        'model_name': 'intfloat/e5-small-v2',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
    },
    'bge-base': {
        'model_name': 'BAAI/bge-base-en-v1.5',
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
    },
}

CHUNK_CONFIGS = {
    256:  {'chunk_size': 256,  'chunk_overlap': 25},
    512:  {'chunk_size': 512,  'chunk_overlap': 50},
    768:  {'chunk_size': 768,  'chunk_overlap': 75},
    1024: {'chunk_size': 1024, 'chunk_overlap': 100},
}

PRIMARY_MODEL = 'bge-base'
PRIMARY_CHUNK_SIZE = 512

GENERATION_MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'
FOMC_DATASET = 'vtasca/fomc-statements-minutes'
