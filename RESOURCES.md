# Game resources: what is on the disc and in RAM

Reverse-engineering notes on the original *Grand Theft Auto* (PlayStation,
NTSC-U `SLUS-00106`), for building ground truth and reward signals from
the game's own data rather than from pixels. Everything here was read off
our own disc image and the running game. The tools below contain no game
data, and everything they extract goes to `scratch/` (gitignored): it is
the game's copyrighted content and is not for redistribution.

Status of each item is marked **verified** (checked against the running
game or an exact round trip) or **partial** (structure known, semantics
not yet).

## Quick start

```bash
.venv/bin/python tools/disc_extract.py            # disc -> scratch/disc/
for f in MISSION NYC MIAMI SANB; do
  .venv/bin/python tools/vsr.py scratch/disc/$f.VSR --out scratch/disc/$f
done
.venv/bin/python tools/gta_text.py                # all text -> scratch/text/
```

| tool | does |
|---|---|
| [tools/disc_extract.py](tools/disc_extract.py) | ISO9660 reader for the MODE2/2352 data track; lists or extracts every file. No external tools. |
| [tools/vsr.py](tools/vsr.py) | Unpacks `.VSR` archives (trees of named chunks). |
| [tools/gta_text.py](tools/gta_text.py) | Decodes every text table, all languages, to `scratch/text/<lang>.json`. |

Disassembly used `capstone` (installed in `.venv`), MIPS32 little endian,
over `SLUS_001.06`.

## The disc (verified)

Track 1 is the data track (MODE2/2352: 2048 bytes of file data at offset
24 of each 2352-byte sector); tracks 2-11 are CD audio. Files:

| file | contents |
|---|---|
| `SLUS_001.06` | the executable. PS-X EXE, 0x800-byte header, loaded at `0x80010000`, entry `0x800f2984`. |
| `MISSION.VSR` | all text (Huffman-compressed) and the six levels' compiled mission scripts |
| `NYC.VSR`, `MIAMI.VSR`, `SANB.VSR` | the three cities (Liberty City, Vice City, San Andreas): map and navigation data |
| `FRONTEND.VSR`, `GAMEFONT.VSR`, `FNT_8X8.VSR`, `PALETTES.VSR`, `LOADING.VSR` | menus, fonts, palettes |
| `AUDIO.VSR` | sound effects |
| `CUT*.STR`, `FULL.STR` | FMV cutscenes |
| `MUSIC/*.WAV` | directory entries pointing into the CD-audio tracks (not data) |

The executable does not name chunks; archives are opened by file name
(`MISSION.VSR`, ...) and chunks by index.

## VSR archives (verified)

```
0x00  "3RSV"  u32 data_size  u32 dir_size  ...  u32 n_entries @0x18
0x24  "SFFO" u32 0  "CRID" u32 dir_bytes ...
0x38  n_entries x 40-byte entries, depth-first tree:
        u32 offset  u32 size  u16 name_id  u16 index  u16 type  u16 0xcccc
        u32 -1 ...  u32 n_children @+0x18
        type 0x16 = folder (n_children entries follow), 0x0e = file
then  "NRSV" ...  "TRTS" u32 bytes  u32 n_names  n_names x u32 offset
        (offsets count from the n_names field), NUL-terminated names
```

Chunk data offsets are relative to `file_size - data_size`.

### MISSION.VSR tree

```
NON_INT/<LANG>        in-game messages shared by all levels
FRONTEND/<LANG>       menu text
HUFF_TABLES/<LANG>    Huffman tree per language
MISSION1/   MISSION (compiled script) + <LANG> (that level's text)   Liberty City, easy
MISSION2/   "                                                     "  Liberty City, medium
MISSION102/ MISSION103/                                              Vice City
MISSION202/ MISSION203/                                              San Andreas
<LANG> = ENGLISH FRENCH GERMAN ITALIAN SPECIAL JAPANESE
```

### City VSRs (NYC / MIAMI / SANB)

`HEADER` (32 B), `BASE` (128 KiB), `COLUMNS`, `BLOCK_INFO`, `OBJECTS`,
`ROUTES`, `LOCATIONS` (108 B), `NAV_DATA`. Not decoded yet. `BASE` +
`COLUMNS` + `BLOCK_INFO` are presumably the 3-D block map (256 x 256
columns, as on PC), and `NAV_DATA` the district areas whose names the HUD
shows.

## Text (verified)

1,909 strings per language, identical ids across languages:

| group | strings | id range | contents |
|---|---|---|---|
| `NON_INT` | 396 | 0-... | tips, HUD messages, district name parts ("Park") |
| `FRONTEND` | 134 | 0-... | menus |
| `MISSION1` / `MISSION2` | 142 / 190 | 1001-, 1200- | Liberty City briefings, pager messages, outcomes |
| `MISSION102` / `MISSION103` | 248 / 215 | 2001- | Vice City |
| `MISSION202` / `MISSION203` | 303 / 281 | 2500-, 3001- | San Andreas |

Fixed ids worth knowing: **2500** `MISSION COMPLETE!`, **2501** `MISSION
FAILED!`, **2503** `KILL FRENZY!`, **2504** `FRENZY FAILED!`, **2505**
`EXTRA LIFE BONUS!`. Ids follow PC GTA1's numbering. District names are
composed from parts at run time ("Southwest" + "Park").

### Text table format

```
u32 index_bytes
index_bytes / 8 x (u32 id, u32 offset)     offsets into the bitstream area
u32 data_bytes
bitstreams                                  one Huffman-coded string each
```

### Huffman scheme

Read from the game's own code: `get_text(table, out, id)` at
**`0x800c9d60`** looks `id` up in the index and calls the per-character
decoder at **`0x8010a5b0`** until it returns 0. The tree loader at
`0x8010a650` takes the first u16 as the root index and copies the rest.

```
HUFF_TABLES/<LANG>:  u32 size, u16 root, u16 table[...]
node word w:   0-bit -> node + (w >> 2)      1-bit -> node + (w >> 2) + 1
               w & 2: the 0-child is a leaf   w & 1: the 1-child is a leaf
               a leaf's low byte is the character
bits are read MSB-first from each byte; character 0 ends the string
```

Round trip check: message 1001 of `MISSION1/ENGLISH` decodes byte for
byte to the briefing the game displays at our `config/boot.state`.

Callers of `get_text` pass the text-table pointer from globals at
`0x8001999c` / `0x800199a0`, an output buffer, and the id; many then
`sprintf` (`0x80121094`) the result into the display buffer.

## Mission scripts (partial)

`MISSION<n>/MISSION` is a compiled form of PC GTA1's `MISSION.INI`:

```
0x00  title, NUL-padded ("Easy Liberty City NEW")
0x18  8 bytes header
0x20  object records, 16 bytes each:
        u16 line   u8 type   u8 flag   u16 x  u16 y  u16 z   u8 ?  u8 kind  6 x 0
        x, y are map-block coordinates (0-255), z the block height
...   command records, 20 bytes each:
        u16 line  u16 p0  u32 p1  u32 p2  u32 p3  u16 p4  u16 opcode
...   a later table of 20-byte records indexed by message id
```

Message ids appear as command parameters (e.g. line 11 of MISSION1 carries
1001), so every message can be tied to the objects and coordinates its
mission uses once the opcodes are named. **Open:** object type numbers
(which is a phone, a car, a target) and the opcode table; both live in the
script interpreter in `SLUS_001.06`.

## RAM (PS1 main RAM, 2 MiB at `0x80000000`)

| address | what | status |
|---|---|---|
| `0x8002C2F0` | score, u32 dollars | verified (in `config/ram_map.json`) |
| `0x8002C96C` / `0x8002C970` | player x / y, 16.16 fixed point | verified (in `config/ram_map.json`) |
| `0x80027FD0` | scratch buffer that `get_text` + `sprintf` fill | verified |
| `0x800444DC` | briefing channel: message on screen (full text) | verified |
| `0x80019FF8` | briefing channel: on-screen flag, u8 | verified (1 while shown, 0 after) |
| `0x8001A004` | briefing channel: scroll countdown, u32 (108 -> 0 over ~330 frames) | verified |
| `0x8001A00C` | briefing channel: message kind, u32 (2 = mission briefing) | verified for kind 2 |
| `0x8004461C` / `0x8004475C` | briefing channel: queued / saved message | seen |
| `0x80045558` | current message split into display lines | seen |
| `0x800448C8` | second ("centre") channel text, filled by `0x800bb288` / `0x800bb540` | from code; not yet seen live |
| `0x8001A034`-`0x8001A044` | centre channel state (-1 when idle) | from code |
| `0x800443CC` | banner: district name, or vehicle name on entering a car | verified |
| `0x8013EE3E` | "Target Score : ..." readout | seen |

`$gp` = `0x80019414` (set at `0x800f298c`); many of the variables above are
`$gp`-relative in the code. All addresses in `0x80010000`-`0x80130000`
belong to the executable's static data and do not move between
savestates.

## Messages in the env (verified for the briefing channel)

The game keeps decoded text, not ids, in its display buffers
(`show_mission_message` around `0x800bab08` decodes with `get_text` and
hands the string to the display routines). Since every string is known,
[gta1env/text.py](gta1env/text.py) maps the text on screen back to its id,
matching `sprintf` templates as patterns. `GTA1Env` adds to `info` every
step:

| key | meaning |
|---|---|
| `message`, `message_on`, `message_kind` | briefing channel |
| `message_id`, `message_group` | its id and table (e.g. 1001, `MISSION1`); -1 / None if unknown |
| `centre_message`, `centre_id`, `centre_group` | second channel |
| `new_message_ids` | ids of messages that appeared on this step (event) |
| `banner` | district or vehicle name |

Config: `read_messages` (default on), `text_table` (default
`scratch/text/english.json`, from `tools/gta_text.py`),
`mission_complete_reward` / `mission_failed_penalty` (default 0, paid
once when 2500 / 2501 appears on either channel).

Checked live: at `config/boot.state` message 1001 is identified, goes off
after ~5.5 s. **Not yet seen live:** any new message appearing mid-play.
The four human recordings never reach a phone, and 10,000 steps of the
160x120 PPO policy triggered no message at all: it never answers a
phone. Confirming the centre channel and 2500 / 2501 needs a recording
of someone answering a phone and finishing (or failing) a job.

## Open work

1. **See a new message live** (phone, pager, mission complete / failed),
   ideally from a `tools/play.py` recording, to confirm the centre
   channel and the reward hooks.
2. **Arrow target in RAM.** The objective position, for a "got closer to
   the objective" reward in place of raw distance; cross-check against the
   script's object coordinates.
3. **Script opcodes and object types.**
4. **City map** (`BASE`/`COLUMNS`/`BLOCK_INFO`) and **districts**
   (`NAV_DATA`), for map-level ground truth.
