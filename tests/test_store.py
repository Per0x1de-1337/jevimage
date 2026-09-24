"""The head directory: a dict that has to agree with the filesystem."""
import pytest
import torch

from jevimage.store import HeadStore, default_dir, valid_name


def record():
    """A head-shaped payload. `tag` is not a real head field - it is here so a reload
    can be shown to bring back the whole record, not just the name and the weights."""
    return {"W": torch.full((16, 2), 0.5), "b": torch.zeros(2),
            "classes": ["red", "blue"], "n": 8, "encoder": "toy", "tag": "a"}


def test_save_then_reload(heads_dir):
    HeadStore(heads_dir).save("colours", record())
    assert (heads_dir / "colours.pt").exists()
    assert not list(heads_dir.glob("*.tmp"))      # the write is atomic, not left half-done

    fresh = HeadStore(heads_dir)
    assert list(fresh) == ["colours"]
    assert fresh["colours"]["classes"] == ["red", "blue"]
    assert fresh["colours"]["encoder"] == "toy"
    assert torch.equal(fresh["colours"]["W"], torch.full((16, 2), 0.5))


def test_a_second_store_sees_what_the_first_wrote(heads_dir):
    """The case a server hits: one worker trains, another has to answer with it."""
    reader = HeadStore(heads_dir)
    assert "colours" not in reader

    HeadStore(heads_dir).save("colours", record())

    assert "colours" in reader                     # __contains__ re-reads the directory
    assert reader["colours"]["tag"] == "a"
    assert reader["colours"]["classes"] == ["red", "blue"]


def test_delete(heads_dir):
    store = HeadStore(heads_dir)
    store.save("colours", record())
    reader = HeadStore(heads_dir)

    assert store.delete("colours") is True
    assert store.delete("colours") is False        # idempotent: no crash on a stale name
    assert not (heads_dir / "colours.pt").exists()
    assert "colours" not in store
    assert "colours" not in reader.sync()


def test_missing_head_says_where_it_looked(heads_dir):
    store = HeadStore(heads_dir)
    store.save("colours", record())
    with pytest.raises(KeyError) as e:
        store["street"]
    assert "street" in str(e.value) and str(heads_dir) in str(e.value)
    assert "colours" in str(e.value)               # names what it does have


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "colours/../..", "",
                                  "a" * 65, ".hidden", "with space", 7, None])
def test_invalid_names_are_refused_not_sanitised(heads_dir, name):
    store = HeadStore(heads_dir)
    for call in (lambda: store.path(name),
                 lambda: store.save(name, record()),
                 lambda: store.delete(name)):
        with pytest.raises(ValueError, match="invalid head name"):
            call()
    assert not list(heads_dir.glob("**/*.pt"))


@pytest.mark.parametrize("name", ["a", "A1", "street_objects", "v1.2-final", "a" * 64])
def test_valid_names(name):
    assert valid_name(name) == name


def test_a_foreign_file_does_not_break_startup(heads_dir):
    heads_dir.mkdir(parents=True)
    (heads_dir / "junk.pt").write_text("not a torch file")
    HeadStore(heads_dir).save("colours", record())

    store = HeadStore(heads_dir)                  # must not raise on the junk
    assert list(store) == ["colours"]
    # but it must not vanish either: "no head named junk" about a file sitting right
    # there in the directory just printed is a lie
    assert store.broken.get("junk")
    with pytest.raises(KeyError, match="present in .* but unreadable"):
        store["junk"]


def test_missing_directory_is_not_an_error(tmp_path):
    store = HeadStore(tmp_path / "never" / "created")
    assert len(store) == 0
    store.save("colours", record())                # created on first write
    assert list(HeadStore(tmp_path / "never" / "created")) == ["colours"]


def test_default_dir_follows_jev_home(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_HOME", str(tmp_path))
    monkeypatch.delenv("JEV_HEADS_DIR", raising=False)
    assert default_dir() == tmp_path / "heads"
    monkeypatch.setenv("JEV_HEADS_DIR", str(tmp_path / "elsewhere"))
    assert default_dir() == tmp_path / "elsewhere"


def test_a_pt_that_loads_but_is_not_a_head_is_broken_not_stored(heads_dir):
    """One of someone else's checkpoints in the directory used to take down the whole
    listing: it entered the store and public() then raised on it."""
    heads_dir.mkdir(parents=True)
    torch.save({"weight": torch.randn(4, 4)}, heads_dir / "checkpoint.pt")
    torch.save(torch.randn(3, 3), heads_dir / "tensoronly.pt")
    HeadStore(heads_dir).save("colours", record())

    store = HeadStore(heads_dir)
    assert list(store) == ["colours"]
    assert "not a jev head" in store.broken["checkpoint"]
    assert "not a jev head" in store.broken["tensoronly"]


def test_a_file_that_cannot_be_named_is_broken_not_listed(heads_dir):
    """`heads` and `ask` used to accept a stem that `rm` refuses, so the CLI listed and
    ran a head it could not delete."""
    heads_dir.mkdir(parents=True)
    HeadStore(heads_dir).save("colours", record())
    (heads_dir / "bad name.pt").write_bytes((heads_dir / "colours.pt").read_bytes())

    store = HeadStore(heads_dir)
    assert list(store) == ["colours"]
    assert "invalid head name" in store.broken["bad name"]


def test_an_unreadable_directory_is_an_error_not_an_empty_listing(heads_dir):
    import os
    heads_dir.mkdir(parents=True)
    HeadStore(heads_dir).save("colours", record())
    os.chmod(heads_dir, 0o000)
    try:
        with pytest.raises(OSError, match="cannot read heads directory"):
            HeadStore(heads_dir)
    finally:
        os.chmod(heads_dir, 0o700)


def test_an_unwritable_directory_is_one_line_not_a_torch_traceback(heads_dir):
    """torch.save re-raises PermissionError as RuntimeError, which is outside the set
    the CLI catches, so `jev train` printed 24 lines of traceback."""
    import os
    heads_dir.mkdir(parents=True)
    os.chmod(heads_dir, 0o500)
    try:
        with pytest.raises(OSError, match="cannot write head 'colours'"):
            HeadStore(heads_dir).save("colours", record())
    finally:
        os.chmod(heads_dir, 0o700)
