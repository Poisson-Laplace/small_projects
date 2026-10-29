#!/home/ubuntu/venvenv/bin/python3
"""
Topolojideki Kahve Kupası -> Torus (Mug to Donut) Homeomorphism / Sürekli Deformasyon Animasyonu:
- Pure Topoloji anlatmak için:
  Genus g = 1 değişmezi
  Euler, chi = 2 - 2g = 0
  Homeomorfizm sürekli yüzey dönüşümü
 Matplotlib 3D + Marching Cubes SDF (Signed Distance Field) + Phong Shading
 Full HD 1920x1080 çözünürlük
 MP4 (H.264) video ve yüksek kaliteli GIF çıktısı
"""

import os
import shutil
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from PIL import Image

PROJECT_DIR = "./"
FRAMES_DIR = "/tmp/mug_to_donut_frames"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


# ==============================================================================
# 3D Grid & SDF Tanımları
# ==============================================================================
GRID_RES = 100
x_coords = np.linspace(-2.1, 2.1, GRID_RES)
y_coords = np.linspace(-1.7, 1.7, GRID_RES)
z_coords = np.linspace(-1.7, 1.7, GRID_RES)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
SPACING = (4.2 / (GRID_RES - 1), 3.4 / (GRID_RES - 1), 3.4 / (GRID_RES - 1))


def smin(a, b, k=0.18):
    """Smooth minimum (yumuşak birleşim)"""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


# 1. Torus Hedefi (Donut)
# x-z düzleminde merkez (0, 0, 0), y ekseni boyunca delik
R_TORUS = 0.95
r_TORUS = 0.40
q_torus = np.hypot(X, Z) - R_TORUS
F_TORUS = np.hypot(q_torus, Y) - r_TORUS

# 2. Katı Kupa Gövdesi (Solid Mug)
# Gövde silindiri
r_out = 0.72
h_body = 0.78
d_cyl_xy = np.hypot(X - 0.45, Y) - r_out
d_cyl_z = np.abs(Z) - h_body
d_cup_outer = np.maximum(d_cyl_xy, d_cyl_z)

# Kulp (Torus yayı x < 0 bölgesinde)
q_xz = np.hypot(X + 0.35, Z) - 0.58
d_handle = np.hypot(q_xz, Y) - 0.20
F_SOLID_MUG = smin(d_cup_outer, d_handle, k=0.18)


def compute_sdf(tau):
    """
    tau in [0.0, 1.0]:
    - tau in [0.0, 0.28]: Kupa içindeki çukur yükselip düzleşsin (iç boşluk yok olur).
    - tau in [0.28, 1.0]: Katı kupa gövdesi bükülerek simit halkasına dönüşür, kulp kalınlaşarak simitin diğer yarısıyla birleşir.
    """
    if tau < 0.28:
        p = tau / 0.28
        # Pürüzsüz geçiş (ease in-out)
        p_smooth = 3 * p**2 - 2 * p**3
        z_bot = -0.58 + p_smooth * 1.40
        r_in = 0.58 * (1.0 - p_smooth)
        d_cav_xy = np.hypot(X - 0.45, Y) - r_in
        d_cav_z = np.maximum(-Z + z_bot, Z - 1.2)
        d_cavity = np.maximum(d_cav_xy, d_cav_z)
        return np.maximum(F_SOLID_MUG, -d_cavity)
    else:
        s = (tau - 0.28) / 0.72
        s_smooth = 3 * s**2 - 2 * s**3
        return (1.0 - s_smooth) * F_SOLID_MUG + s_smooth * F_TORUS


# ==============================================================================
# Kare Render Fonksiyonu
# ==============================================================================
def render_frame(frame_idx, total_frames, tau, camera_azim, camera_elev):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#080C14")

    # SDF hesapla ve Marching Cubes ile yüzey çıkar
    F = compute_sdf(tau)
    verts, faces, normals, values = marching_cubes(F, level=0, spacing=SPACING)
    verts[:, 0] -= 2.1
    verts[:, 1] -= 1.7
    verts[:, 2] -= 1.7

    # Yüzey normalleri
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norm[fn_norm == 0] = 1.0
    fn = fn / fn_norm

    # Işıklandırma (Lambertian diffuse + Blinn-Phong specular)
    light = np.array([0.65, -0.60, 0.75])
    light /= np.linalg.norm(light)
    diffuse = np.maximum(0, np.sum(fn * light, axis=1))

    view_dir = np.array([0.5, 0.5, 0.7])
    view_dir /= np.linalg.norm(view_dir)
    half_vec = (light + view_dir)
    half_vec /= np.linalg.norm(half_vec)
    specular = np.maximum(0, np.sum(fn * half_vec, axis=1)) ** 16
    intensity = np.clip(0.30 + 0.62 * diffuse + 0.24 * specular, 0, 1.0)

    # Renk geçişi: Sırlı Terracotta Kupasından Altın Sarısı Simite (Donut)
    col_mug = np.array([0.96, 0.38, 0.22])   # Seramik pişmiş toprak
    col_donut = np.array([0.90, 0.54, 0.18]) # Fırınlanmış hamur / simit
    base_col = (1.0 - tau) * col_mug + tau * col_donut
    colors = np.outer(intensity, base_col)
    colors = np.clip(colors, 0, 1.0)

    # Pure white background, zero text
    fig.patch.set_facecolor("#FFFFFF")

    # 3D Axes centered across full canvas
    ax3d = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax3d.set_facecolor("#FFFFFF")
    mesh = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
    ax3d.add_collection3d(mesh)

    ax3d.set_xlim(-1.25, 1.25)
    ax3d.set_ylim(-1.25, 1.25)
    ax3d.set_zlim(-1.25, 1.25)
    ax3d.axis("off")
    ax3d.view_init(elev=camera_elev, azim=camera_azim)

    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(frame_path, facecolor="#FFFFFF", dpi=100)
    plt.close(fig)
    return frame_path


# ==============================================================================
# Ana Animasyon Döngüsü
# ==============================================================================
def generate_animation():
    print("Topoloji Mug to Donut Animasyonu Başlatılıyor...")
    total_frames = 120
    fps = 30

    # tau zaman çizelgesi (ping-pong / ileri-geri döngü)
    # 0..12: Kupa duruşu (tau = 0.0)
    # 12..50: 1. Aşama Kupa içi düzleşme (tau: 0.0 -> 0.28)
    # 50..88: 2. ve 3. Aşama Simite dönüşüm (tau: 0.28 -> 1.0)
    # 88..102: Donut duruşu (tau = 1.0)
    # 102..120: Kupaya yumuşak geri dönüş (kesintisiz loop için)
    
    taus = np.zeros(total_frames)
    for i in range(total_frames):
        if i < 12:
            taus[i] = 0.0
        elif i < 50:
            # 0.0 -> 0.28
            u = (i - 12) / 38.0
            taus[i] = 0.28 * (3 * u**2 - 2 * u**3)
        elif i < 88:
            # 0.28 -> 1.0
            u = (i - 50) / 38.0
            taus[i] = 0.28 + 0.72 * (3 * u**2 - 2 * u**3)
        elif i < 102:
            taus[i] = 1.0
        else:
            # 1.0 -> 0.0 (smooth return for infinite loop)
            u = (i - 102) / 18.0
            taus[i] = 1.0 - (3 * u**2 - 2 * u**3)

    # Kamera açısı: Yumuşak turntable dönüşü (30 derece -> 75 derece)
    azims = 32.0 + 42.0 * (np.arange(total_frames) / float(total_frames))
    elevs = 26.0 + 4.0 * np.sin(2 * np.pi * np.arange(total_frames) / float(total_frames))

    print(f"Toplam {total_frames} kare render edilecek...")
    t_start = time.time()

    sample_frames = {}
    for i in range(total_frames):
        t0 = time.time()
        fpath = render_frame(i, total_frames, taus[i], azims[i], elevs[i])
        t_el = time.time() - t0
        if i % 15 == 0 or i == total_frames - 1:
            print(f"  [Kare {i+1:03d}/{total_frames}] tau={taus[i]:.2f}, süre={t_el:.2f}s")
        
        # Numune kareleri kaydet
        if i == 5:
            sample_frames["start_mug"] = fpath
        elif i == 45:
            sample_frames["stage1_filled"] = fpath
        elif i == 70:
            sample_frames["stage2_morphing"] = fpath
        elif i == 95:
            sample_frames["stage3_donut"] = fpath

    t_render = time.time() - t_start
    print(f"\n Tüm kareler render edildi! Toplam süre: {t_render:.1f} saniye ({t_render/total_frames:.2f}s/kare).")

    # ==============================================================================
    # Video Kodlama (MP4 ve GIF)
    # ==============================================================================
    mp4_path = os.path.join(PROJECT_DIR, "topoloji_mug_to_donut_animasyonu.mp4")
    gif_path = os.path.join(PROJECT_DIR, "topoloji_mug_to_donut_animasyonu.gif")

    print("\n FFMPEG ile MP4 kodlanıyor (Sağdan ve soldan kırpılmış 1080x1080 kare format)...")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(FRAMES_DIR, "frame_%04d.png"),
        "-vf", "crop=1080:1080:420:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    shutil.copy2(mp4_path, os.path.join(ARTIFACT_DIR, "topoloji_mug_to_donut_animasyonu.mp4"))
    print(f"MP4 Başarıyla Kaydedildi: {mp4_path}")

    print("\n GIF Üretiliyor ve Kodlanıyor...")
    palette_path = "/tmp/mug_palette.png"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-vf", "fps=20,scale=720:-1:flags=lanczos,palettegen=stats_mode=diff",
        palette_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-i", palette_path,
        "-lavfi", "fps=20,scale=720:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
        gif_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    shutil.copy2(gif_path, os.path.join(ARTIFACT_DIR, "topoloji_mug_to_donut_animasyonu.gif"))
    print(f"GIF Başarıyla Kaydedildi: {gif_path}")

    # Numune kareleri de kırpıp proje ve artifact dizinine kopyala
    for name, src in sample_frames.items():
        dst_proj = os.path.join(PROJECT_DIR, f"topoloji_mug_to_donut_{name}.png")
        dst_art = os.path.join(ARTIFACT_DIR, f"topoloji_mug_to_donut_{name}.png")
        img_s = Image.open(src)
        img_cropped = img_s.crop((420, 0, 1500, 1080))
        img_cropped.save(dst_proj)
        shutil.copy2(dst_proj, dst_art)

    print("\n TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI! ")


if __name__ == "__main__":
    generate_animation()
