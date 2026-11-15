import collections

import pyarrow.parquet as pq

from zerre.text import phonemize

unknown = collections.Counter()
examples = {}
rows = 0
for batch in pq.ParquetFile("data/tr/pilot.parquet").iter_batches(batch_size=32):
    for row in batch.to_pylist():
        rows += 1
        phones = phonemize(row["text"])
        for phone in phones.unknown:
            unknown[phone] += 1
            examples.setdefault(phone, row["text"][:70])

print(f"rows: {rows}, rows with unknowns: {sum(1 for _ in [])}")
print("unknown phones:", dict(unknown))
for phone, text in examples.items():
    print(f"  {phone!r} <- {text}")
