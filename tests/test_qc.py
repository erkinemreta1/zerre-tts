from zerre.qc import plan, text_hash


def test_plan_keeps_untouched_rows_and_reports_missing_ones():
    manifest = [
        {"id": "a", "text": "Bir cümle."},
        {"id": "b", "text": "Yanlış yazılmış cümle."},
        {"id": "c", "text": "Atılacak cümle."},
    ]
    qc = [
        {"id": "a", "text": "Bir cümle.", "qc": "keep"},
        {"id": "b", "text": "Doğru yazılmış cümle.", "qc": "asr-text"},
    ]
    updates, dropped = plan(manifest, qc)
    assert updates == {"b": "Doğru yazılmış cümle."}
    assert dropped == ["c"]


def test_plan_ignores_empty_asr_text():
    manifest = [{"id": "a", "text": "Bir cümle."}]
    qc = [{"id": "a", "text": "   ", "qc": "asr-text"}]
    updates, dropped = plan(manifest, qc)
    assert updates == {}
    assert dropped == []


def test_text_hash_is_stable_and_strips_whitespace():
    assert text_hash("  merhaba ") == text_hash("merhaba")
    assert text_hash("merhaba") != text_hash("Merhaba")
