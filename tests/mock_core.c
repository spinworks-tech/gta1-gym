/* A fake libretro core, used only to test this repo's plumbing.
 *
 * It implements enough of the libretro API to exercise libretro.py,
 * env.py, vec.py, the RAM scanner and the PPO loop without needing the
 * real BIOS/disc: a 320x240 XRGB8888 frame, a 2 MiB "main RAM" with a
 * cash counter and a health byte at fixed addresses, and a trivial
 * learnable task (drive the white block onto the grey pad to earn cash).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define W 320
#define H 240
#define RAM_SIZE (2 * 1024 * 1024)
#define OFF_CASH   0x0A1234
#define OFF_HEALTH 0x0B5678
#define OFF_POSX   0x0C0000
#define OFF_POSY   0x0C0004
#define OFF_OVER   0x0C0010

enum { PIX_XRGB8888 = 1 };
enum { ENV_SET_PIXEL_FORMAT = 10, ENV_GET_AV_ENABLE = 47 };
enum { ID_B=0, ID_UP=4, ID_DOWN=5, ID_LEFT=6, ID_RIGHT=7, ID_A=8 };

typedef int  (*env_t)(unsigned, void *);
typedef void (*video_t)(const void *, unsigned, unsigned, size_t);
typedef void (*audio_t)(int16_t, int16_t);
typedef size_t (*audio_batch_t)(const int16_t *, size_t);
typedef void (*poll_t)(void);
typedef int16_t (*input_t)(unsigned, unsigned, unsigned, unsigned);

static env_t cb_env;
static video_t cb_video;
static poll_t cb_poll;
static input_t cb_input;
static audio_t cb_audio;
static audio_batch_t cb_audio_batch;

static uint8_t *ram;
static uint32_t *fb;
static int32_t px, py, pad_x, pad_y;
static uint32_t rngstate = 12345;
static uint64_t frames;

static uint32_t rnd(void) {
  rngstate = rngstate * 1103515245u + 12345u;
  return (rngstate >> 16);
}

void retro_set_environment(env_t cb) {
  cb_env = cb;
  unsigned fmt = PIX_XRGB8888;
  cb(ENV_SET_PIXEL_FORMAT, &fmt);
}
void retro_set_video_refresh(video_t cb) { cb_video = cb; }
void retro_set_audio_sample(audio_t cb) { cb_audio = cb; }
void retro_set_audio_sample_batch(audio_batch_t cb) { cb_audio_batch = cb; }
void retro_set_input_poll(poll_t cb) { cb_poll = cb; }
void retro_set_input_state(input_t cb) { cb_input = cb; }

unsigned retro_api_version(void) { return 1; }

void retro_get_system_info(void *info) {
  struct si { const char *name, *ver, *ext; char full, block; } *s = info;
  s->name = "MockCore";
  s->ver = "1.0";
  s->ext = "mock";
  s->full = 0;
  s->block = 0;
}

void retro_get_system_av_info(void *info) {
  struct av { unsigned bw, bh, mw, mh; float ar; double fps, sr; } *a = info;
  a->bw = W; a->bh = H; a->mw = W; a->mh = H; a->ar = 4.0f / 3.0f;
  a->fps = 60.0; a->sr = 44100.0;
}

void retro_init(void) {
  ram = calloc(1, RAM_SIZE);
  fb = calloc(W * H, sizeof(uint32_t));
}
void retro_deinit(void) { free(ram); free(fb); ram = NULL; fb = NULL; }

static void place_pad(void) {
  pad_x = 20 + (int32_t)(rnd() % (W - 60));
  pad_y = 20 + (int32_t)(rnd() % (H - 60));
}

static void reset_state(void) {
  memset(ram, 0, RAM_SIZE);
  px = W / 2; py = H / 2;
  frames = 0;
  rngstate = 12345;
  place_pad();
  ram[OFF_HEALTH] = 100;
  *(uint32_t *)(ram + OFF_CASH) = 0;
}

int retro_load_game(const void *game) {
  (void)game;
  reset_state();
  return 1;
}
void retro_unload_game(void) {}
void retro_reset(void) { reset_state(); }

void retro_set_controller_port_device(unsigned p, unsigned d) { (void)p; (void)d; }
unsigned retro_get_region(void) { return 0; }
int retro_load_game_special(unsigned a, const void *b, size_t c) {
  (void)a; (void)b; (void)c; return 0;
}
void *retro_get_memory_data(unsigned id) { return id == 2 ? ram : NULL; }
size_t retro_get_memory_size(unsigned id) { return id == 2 ? RAM_SIZE : 0; }

/* Savestate = the whole mutable state, so resets are exact. */
struct state { uint8_t ram[RAM_SIZE]; int32_t px, py, pad_x, pad_y;
               uint32_t rngstate; uint64_t frames; };
size_t retro_serialize_size(void) { return sizeof(struct state); }
int retro_serialize(void *data, size_t size) {
  if (size < sizeof(struct state)) return 0;
  struct state *s = data;
  memcpy(s->ram, ram, RAM_SIZE);
  s->px = px; s->py = py; s->pad_x = pad_x; s->pad_y = pad_y;
  s->rngstate = rngstate; s->frames = frames;
  return 1;
}
int retro_unserialize(const void *data, size_t size) {
  if (size < sizeof(struct state)) return 0;
  const struct state *s = data;
  memcpy(ram, s->ram, RAM_SIZE);
  px = s->px; py = s->py; pad_x = s->pad_x; pad_y = s->pad_y;
  rngstate = s->rngstate; frames = s->frames;
  return 1;
}

void retro_cheat_reset(void) {}
void retro_cheat_set(unsigned a, int b, const char *c) { (void)a; (void)b; (void)c; }

static void draw_box(int32_t cx, int32_t cy, int32_t half, uint32_t color) {
  for (int32_t y = cy - half; y <= cy + half; y++) {
    if (y < 0 || y >= H) continue;
    for (int32_t x = cx - half; x <= cx + half; x++) {
      if (x < 0 || x >= W) continue;
      fb[y * W + x] = color;
    }
  }
}

void retro_run(void) {
  cb_poll();
  int16_t up = cb_input(0, 1, 0, ID_UP), down = cb_input(0, 1, 0, ID_DOWN);
  int16_t left = cb_input(0, 1, 0, ID_LEFT), right = cb_input(0, 1, 0, ID_RIGHT);
  int16_t fire = cb_input(0, 1, 0, ID_A);

  py += (down ? 3 : 0) - (up ? 3 : 0);
  px += (right ? 3 : 0) - (left ? 3 : 0);
  if (px < 8) px = 8;
  if (px > W - 9) px = W - 9;
  if (py < 8) py = 8;
  if (py > H - 9) py = H - 9;

  /* Reaching the pad pays out; firing wastes health. */
  int32_t dx = px - pad_x, dy = py - pad_y;
  if (dx * dx + dy * dy < 15 * 15) {
    *(uint32_t *)(ram + OFF_CASH) += 100;
    place_pad();
  }
  frames++;
  if (fire && ram[OFF_HEALTH] > 0) ram[OFF_HEALTH]--;
  if ((frames % 180) == 0 && ram[OFF_HEALTH] > 0) ram[OFF_HEALTH]--;
  ram[OFF_OVER] = (ram[OFF_HEALTH] == 0);
  *(int32_t *)(ram + OFF_POSX) = px;
  *(int32_t *)(ram + OFF_POSY) = py;

  for (int i = 0; i < W * H; i++) fb[i] = 0x00202020;
  draw_box(pad_x, pad_y, 10, 0x00808080);
  draw_box(px, py, 8, 0x00FFFFFF);
  cb_video(fb, W, H, W * sizeof(uint32_t));
  if (cb_audio_batch) {
    static int16_t silence[2 * 735];
    cb_audio_batch(silence, 735);
  }
}
