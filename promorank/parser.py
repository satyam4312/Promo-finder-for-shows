"""Parser for timestamped screenplay text files."""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path


BLOCK_RE = re.compile(
    r"\*\*(?P<label>[^*\n]+):\*\*\s*\n"
    r"Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)\s*\n"
    r"(?P<text>.*?)(?=\n\s*\n\*\*[^*\n]+:\*\*\s*\nStart:|\nSCENE\s+\d+|\n--- SCRIPT END ---|\Z)",
    re.S,
)
SCENE_RE = re.compile(
    r"SCENE\s+(?P<number>\d+)\s*\n"
    r"Scene heading:\s*(?P<heading>.*?)\s*\n"
    r"Scene start:\s*(?P<start>[^\n]+)\s*\n"
    r"Scene end:\s*(?P<end>[^\n]+)",
    re.I,
)
# Every "**LABEL:**" header line, whether or not a valid timestamp line follows it.
HEADER_RE = re.compile(r"^\*\*[^*\n]+:\*\*\s*$", re.M)


@dataclass
class Event:
    episode: int
    scene: int
    scene_heading: str
    label: str
    speaker: str
    event_type: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Episode:
    number: int
    filename: str
    logline: str
    continuity_notes: str
    duration: float
    scenes: list[dict]
    events: list[Event]
    parse_warnings: list[str] = field(default_factory=list)


def parse_timestamp(value: str) -> float:
    """Convert MM:SS.ss (or HH:MM:SS.ss) to seconds."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp: {value!r}")


def _header_value(text: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, re.M | re.I)
    return match.group(1).strip() if match else ""


def parse_episode_text(text: str, filename: str = "episode.txt") -> Episode:
    number_match = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text, re.I)
    if not number_match:
        number_match = re.search(r"episode[-_ ]?(\d+)", filename, re.I)
    if not number_match:
        raise ValueError(f"Could not identify episode number in {filename}")
    episode_number = int(number_match.group(1))

    scenes = []
    for match in SCENE_RE.finditer(text):
        scenes.append({
            "number": int(match.group("number")),
            "heading": match.group("heading").strip(),
            "start": parse_timestamp(match.group("start")),
            "end": parse_timestamp(match.group("end")),
            "position": match.start(),
        })

    raw_events = []
    generic_labels = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound_fx", "SFX": "sound_fx", "SOUND": "sound_fx"}
    matched_blocks = 0
    skipped_blocks = 0
    for match in BLOCK_RE.finditer(text):
        matched_blocks += 1
        label = match.group("label").strip()
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        body = match.group("text").strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if end < start:
            skipped_blocks += 1
            continue
        scene = next((s for s in reversed(scenes) if s["position"] < match.start()), None)
        scene_num = scene["number"] if scene else 0
        scene_heading = scene["heading"] if scene else ""
        base_label = label.split("(", 1)[0].strip().upper()  # "ACTION (CONT'D)" -> "ACTION"
        event_type = generic_labels.get(base_label, "dialogue")
        speaker = label.split("(", 1)[0].strip() if event_type == "dialogue" else ""
        # Script text is supplied as quoted dialogue or prose. Keep wording intact.
        raw_events.append(Event(
            episode=episode_number,
            scene=scene_num,
            scene_heading=scene_heading,
            label=label,
            speaker=speaker,
            event_type=event_type,
            start=start,
            end=end,
            text=body,
        ))

    raw_events.sort(key=lambda event: (event.start, event.end, event.event_type))
    max_scene_end = max((s["end"] for s in scenes), default=0.0)
    max_event_end = max((e.end for e in raw_events), default=0.0)
    duration = max(max_scene_end, max_event_end)
    scenes_clean = [{k: v for k, v in s.items() if k != "position"} for s in scenes]

    # A format change in a new show would otherwise silently produce few or no events.
    warnings = []
    header_count = len(HEADER_RE.findall(text))
    if header_count != matched_blocks:
        warnings.append(
            f"{Path(filename).name}: found {header_count} **LABEL:** headers but parsed "
            f"{matched_blocks} timestamped blocks; check the script format."
        )
    if skipped_blocks:
        warnings.append(f"{Path(filename).name}: dropped {skipped_blocks} block(s) whose end is before their start.")
    if not raw_events:
        warnings.append(f"{Path(filename).name}: no timestamped blocks were parsed.")
    if not scenes:
        warnings.append(f"{Path(filename).name}: no SCENE headers were parsed.")

    return Episode(
        number=episode_number,
        filename=Path(filename).name,
        logline=_header_value(text, "EPISODE LOGLINE"),
        continuity_notes=_header_value(text, "CONTINUITY NOTES"),
        duration=duration,
        scenes=scenes_clean,
        events=raw_events,
        parse_warnings=warnings,
    )


def load_zip_bytes(zip_bytes: bytes) -> list[Episode]:
    """Load and parse all .txt screenplay files from an uploaded ZIP."""
    import io

    episodes: list[Episode] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith(".txt") and not name.endswith("/")
            # Skip macOS resource-fork junk such as __MACOSX/._episode-001.txt
            and not name.startswith("__MACOSX/") and not Path(name).name.startswith("._")
        ]
        if not names:
            raise ValueError("The ZIP contains no .txt screenplay files.")
        for name in names:
            raw = archive.read(name)
            decoded = None
            for encoding in ("utf-8-sig", "utf-8", "cp1252"):
                try:
                    decoded = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded is None:
                decoded = raw.decode("utf-8", errors="replace")
            episodes.append(parse_episode_text(decoded, name))
    seen: dict[int, str] = {}
    for episode in episodes:
        if episode.number in seen:
            raise ValueError(
                f"Episode {episode.number} appears twice: {seen[episode.number]} and {episode.filename}."
            )
        seen[episode.number] = episode.filename
    return sorted(episodes, key=lambda episode: episode.number)


def load_zip_path(path: str | Path) -> list[Episode]:
    return load_zip_bytes(Path(path).read_bytes())
