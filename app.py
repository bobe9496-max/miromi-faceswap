# app.py — Miromi Wedding Face Swap (Face Detail Enhanced)
# - 모델 로드: 로컬(LFS) → 홈캐시 → custom URL → InsightFace 릴리스
# - 성별 고려 자동 매핑 + 수동 오버라이드
# - 단일 소스 2인 자동 추출
# - 사전 업스케일(SSAA) + 선택적 사후 업스케일
# - 피부톤(Reinhard) + Poisson 블렌딩
# - NEW: 얼굴 패치 전용 디테일 업스케일 (Real-ESRGAN → OpenCV 폴백)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import requests
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# --- (선택) Real-ESRGAN 엔진 준비 ---
def build_realesrganer():
    try:
        from realesrgan import RealESRGANer
        # 모델 자동 다운로드(가끔 차단될 수 있음) → 실패 시 except
        return RealESRGANer(
            scale=4,
            model_path=None,                  # 내부 기본 경로 사용(자동 다운로드)
            model="RealESRGAN_x4plus",
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True
        )
    except Exception:
        return None

REALSER_AVAILABLE = build_realesrganer() is not None  # 한번 시도해서 가능 여부만 체크

st.set_page_config(page_title="Miromi Wedding Face Swap (2 faces)", layout="wide")


# ----------------------------- Utils -----------------------------
def read_image(file):
    if hasattr(file, "read"):
        data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img

def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clamp_rect(x1,y1,x2,y2,w,h):
    x1 = max(0, min(int(x1), w-1)); x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h-1)); y2 = max(0, min(int(y2), h))
    if x2 <= x1: x2 = min(w, x1+1)
    if y2 <= y1: y2 = min(h, y1+1)
    return x1,y1,x2,y2

def crop_face(img_bgr, face, pad=0.25):
    h, w = img_bgr.shape[:2]
    x1,y1,x2,y2 = map(int, face.bbox)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    bw2, bh2 = int(bw*(1+pad)), int(bh*(1+pad*1.2))
    x1n, x2n = int(cx - bw2/2), int(cx + bw2/2)
    y1n, y2n = int(cy - bh2/2), int(cy + bh2/2)
    x1n,y1n,x2n,y2n = clamp_rect(x1n,y1n,x2n,y2n,w,h)
    return img_bgr[y1n:y2n, x1n:x2n]

def cosine_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def draw_faces_preview(img_bgr, faces, color=(0,255,0)):
    vis = img_bgr.copy()
    for idx, f in enumerate(faces):
        x1,y1,x2,y2 = map(int, f.bbox)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"#{idx}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis


# ---------------------- Gender-aware mapping ----------------------
def get_sex(face):
    val = getattr(face, "sex", None)
    if val is None: val = getattr(face, "gender", None)
    if isinstance(val, (int, float)):
        return "M" if val == 1 else ("F" if val == 0 else None)
    if isinstance(val, str):
        v = val.lower()
        if "m" in v: return "M"
        if "f" in v: return "F"
    return None

def gender_label(x): return {"M":"남","F":"여", None:"불명"}[x]

def map_sources_to_targets_gender_aware(src_faces, tgt_faces, src_feats, tgt_feats, gender_penalty=0.35):
    m, n = len(src_faces), len(tgt_faces)
    assert m in (1,2)
    src_g = [get_sex(f) for f in src_faces]
    tgt_g = [get_sex(f) for f in tgt_faces]
    sims = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            sims[i, j] = cosine_sim(src_feats[i], tgt_feats[j])

    def cost(pairing):
        c = 0.0
        for (i, j) in pairing:
            c += -sims[i, j]
            if src_g[i] is not None and tgt_g[j] is not None and src_g[i] != tgt_g[j]:
                c += gender_penalty
        return c

    if m == 1:
        best = min(range(n), key=lambda j: cost([(0, j)]))
        return [(0, best, float(sims[0, best]))], sims

    import itertools
    best_pair, best_cost, best_sims = None, 1e9, None
    for j0, j1 in itertools.permutations(range(n), 2):
        pr = [(0, j0), (1, j1)]
        c = cost(pr)
        if c < best_cost:
            best_cost, best_pair = c, pr
            best_sims = (sims[0, j0], sims[1, j1])
    mapping = [(best_pair[0][0], best_pair[0][1], float(best_sims[0])),
               (best_pair[1][0], best_pair[1][1], float(best_sims[1]))]
    return mapping, sims


# -------------------- Harmonization / Detail ----------------------
def reinhard_transfer(src_bgr, ref_bgr):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        s_mean, s_std = src[:,:,ch].mean(), src[:,:,ch].std() + 1e-6
        r_mean, r_std = ref[:,:,ch].mean(), ref[:,:,ch].std() + 1e-6
        src[:,:,ch] = (src[:,:,ch] - s_mean) * (r_std / s_std) + r_mean
    return cv2.cvtColor(np.clip(src,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def face_mask_from_wh(width, height, feather=0.15):
    mask = np.zeros((height, width), np.uint8)
    cx, cy = width//2, height//2
    rx, ry = int(width*0.5), int(height*0.5)
    cv2.ellipse(mask, (cx,cy), (rx,ry), 0, 0, 360, 255, -1)
    if feather>0:
        k = int(max(3, (width+height)*0.5*feather));  k += (k%2==0)
        mask = cv2.GaussianBlur(mask, (k,k), 0)
    return mask

def unsharp_mask(img, radius=1.2, amount=0.8):
    blur = cv2.GaussianBlur(img, (0,0), radius)
    return cv2.addWeighted(img, 1+amount, blur, -amount, 0)

# --- NEW: 얼굴 패치 디테일 강화 ---
def enhance_face_patch(patch_bgr, engine="auto", strength="Strong"):
    """
    engine: 'auto' | 'opencv' | 'realesrgan'
    strength: 'Light' | 'Medium' | 'Strong'
    """
    h, w = patch_bgr.shape[:2]
    if h < 8 or w < 8:
        return patch_bgr

    if engine == "auto":
        use_realesr = REALSER_AVAILABLE
    elif engine == "realesrgan":
        use_realesr = REALSER_AVAILABLE
    else:
        use_realesr = False

    # 1) Real-ESRGAN 경로 (가능할 때만)
    if use_realesr:
        try:
            from realesrgan import RealESRGANer
            # 새로 열어야 안정적인 경우가 있어 재생성
            sr = RealESRGANer(scale=4, model="RealESRGAN_x4plus", tile=0, tile_pad=10, pre_pad=0, half=True)
            scale_map = {"Light": 2.0, "Medium": 3.0, "Strong": 4.0}
            s = scale_map.get(strength, 3.0)
            up, _ = sr.enhance(patch_bgr, outscale=s)
            # 원크기로 복구(세부 감 유지)
            out = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)
            out = unsharp_mask(out, radius=1.0, amount=0.25)
            return out
        except Exception:
            # 실패 시 OpenCV 파이프라인으로 폴백
            pass

    # 2) OpenCV 경량 파이프라인
    # bilateral로 노이즈 낮추고, detailEnhance로 질감 살린 뒤, 언샵
    iters = {"Light":1, "Medium":2, "Strong":3}.get(strength, 2)
    out = patch_bgr.copy()
    for _ in range(iters):
        out = cv2.bilateralFilter(out, d=0, sigmaColor=30, sigmaSpace=7)
        out = cv2.detailEnhance(out, sigma_s=10, sigma_r=0.15)
        out = unsharp_mask(out, radius=1.1, amount=0.5)
    return out


# ------------------------- Model loader --------------------------
def _download_to(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    return path

def get_inswapper(providers):
    local = Path("models/inswapper_128.onnx")
    cache = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"
    custom_url = ""
    try:
        custom_url = (st.secrets.get("INSWAPPER_URL") or "").strip()
    except Exception:
        pass
    custom_url = (custom_url or os.getenv("INSWAPPER_URL", "")).strip()

    if local.exists() and local.stat().st_size > 10_000_000:
        st.info(f"Using bundled model: {local}")
        return get_model(str(local), providers=providers)
    if cache.exists() and cache.stat().st_size > 10_000_000:
        st.info(f"Using cached model: {cache}")
        return get_model(str(cache), providers=providers)
    if custom_url:
        st.warning("Downloading inswapper_128.onnx from INSWAPPER_URL …")
        try:
            _download_to(cache, custom_url)
            st.success("Download complete.")
            return get_model(str(cache), providers=providers)
        except Exception as e:
            st.error(f"Custom URL download failed: {e}")
    st.warning("Falling back to InsightFace release download (may fail on Cloud).")
    return get_model("inswapper_128.onnx", providers=providers, download=True, download_zip=True)

@st.cache_resource(show_spinner="모델 로딩 중…")
def load_models(use_gpu=False, det_size=(640,640)):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=(0 if use_gpu else -1), det_size=det_size)
    swapper = get_inswapper(providers)
    return app, swapper


# ------------------------------ UI ------------------------------
st.title("Miromi Wedding Face Swap")
st.caption("2인 스왑 • 성별 고려 매핑 • 디테일 업스케일")

with st.sidebar:
    st.header("세팅")
    gpu = st.checkbox("GPU 사용 (CUDA)", value=False)
    det = st.select_slider("검출 해상도", [320, 480, 640, 800, 960], value=800)

    st.subheader("해상도 / 품질")
    pre_scale = float(st.select_slider("사전 업스케일(SSAA)", [1.0, 1.25, 1.5, 1.75, 2.0], value=1.5))
    keep_prescaled = st.checkbox("최종 해상도: 사전 업스케일 유지", value=True)
    post_scale = float(st.select_slider("사후 업스케일", [1.0, 1.25, 1.5, 1.75, 2.0], value=1.0))

    st.subheader("보정 옵션")
    keep_color = st.checkbox("피부톤 동기화 (Reinhard)", value=True)
    use_poisson = st.checkbox("경계 블렌딩 (Poisson)", value=True)
    detail_boost = float(st.slider("언샵 강도(기본 보정)", 0.0, 1.2, 0.4, 0.1))
    use_clahe = st.checkbox("CLAHE(명암 디테일)", value=False)

    st.subheader("얼굴 디테일 업스케일")
    engine = st.selectbox("엔진", ["auto", "realesrgan", "opencv"], index=0,
                          help="auto: Real-ESRGAN 설치 시 사용, 없으면 OpenCV 보정")
    face_sr = st.select_slider("강도", ["Off", "Light", "Medium", "Strong"], value="Medium")
    if face_sr == "Off":
        engine = "opencv"  # 사실상 끔

    # 비상용: 모델 수동 업로드
    up = st.file_uploader("모델 수동 업로드(.onnx)", type=["onnx"])
    if up:
        Path("models").mkdir(exist_ok=True)
        with open("models/inswapper_128.onnx", "wb") as f:
            f.write(up.getbuffer())
        st.success("모델 저장 완료! Rerun(F5) 해주세요.")

    app, swapper = load_models(use_gpu=gpu, det_size=(det, det))

st.subheader("1) 소스 모드")
mode = st.radio("소스를 어떻게 올릴까요?", ["개별 업로드 (A/B)", "한 장에서 자동 2명"])

src_files, multi_img = [], None
if mode == "개별 업로드 (A/B)":
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("소스 A", type=["jpg","jpeg","png"], key="srcA")
        if f1: src_files.append(("A", f1))
    with c2:
        f2 = st.file_uploader("소스 B (선택)", type=["jpg","jpeg","png"], key="srcB")
        if f2: src_files.append(("B", f2))
else:
    multi_img = st.file_uploader("두 사람이 함께 있는 단일 소스 이미지", type=["jpg","jpeg","png"], key="srcBoth")

st.subheader("2) 타겟 웨딩 사진 업로드")
tfile = st.file_uploader("타겟", type=["jpg","jpeg","png"], key="target")

run = st.button("얼굴 스왑 실행", type="primary", use_container_width=True)


# ----------------------------- Main -----------------------------
if run:
    sources = []

    # 소스 준비
    if mode == "개별 업로드 (A/B)":
        if len(src_files) == 0:
            st.error("소스 얼굴을 최소 1개 업로드해 주세요."); st.stop()
        for label, f in src_files:
            img = read_image(f)
            sfaces = app.get(img)
            if len(sfaces) == 0:
                st.error(f"{label}에서 얼굴을 찾지 못했습니다."); st.stop()
            areas = [ (sf.bbox[2]-sf.bbox[0])*(sf.bbox[3]-sf.bbox[1]) for sf in sfaces ]
            s_pick = sfaces[int(np.argmax(areas))]
            sources.append((label, img, s_pick, getattr(f, "name", label)))
    else:
        if multi_img is None:
            st.error("단일 소스 이미지를 업로드해 주세요."); st.stop()
        img = read_image(multi_img)
        sfaces = app.get(img)
        if len(sfaces) < 2:
            st.error(f"단일 소스에서 {len(sfaces)}명만 감지됨 — 2명이 보이도록 다시 올려주세요."); st.stop()
        idxs = np.argsort([ (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in sfaces ])[::-1][:2]
        two = [sfaces[i] for i in idxs]
        two = sorted(two, key=lambda f: f.bbox[0])  # 좌→우
        sources = [("A", img, two[0], getattr(multi_img,"name","A_from_single")),
                   ("B", img, two[1], getattr(multi_img,"name","B_from_single"))]

    # 소스 프리뷰
    st.markdown("**소스 얼굴 프리뷰**")
    cols = st.columns(2)
    for i, s in enumerate(sources[:2]):
        label, simg, sface, fname = s
        with cols[i]:
            st.write(f"{label} — `{fname}` — 성별: {gender_label(get_sex(sface))}")
            st.image(bgr2rgb(crop_face(simg, sface)), caption=f"Source {label}", use_container_width=True)

    # 타겟 + 사전 업스케일
    if tfile is None:
        st.error("타겟 웨딩 사진을 업로드해 주세요."); st.stop()
    tgt_orig = read_image(tfile)
    oh, ow = tgt_orig.shape[:2]
    tgt_for_detect = cv2.resize(tgt_orig, (int(ow*pre_scale), int(oh*pre_scale)), interpolation=cv2.INTER_CUBIC) if pre_scale>1.0 else tgt_orig.copy()

    tgt_faces = app.get(tgt_for_detect)
    if len(tgt_faces) == 0:
        st.error("타겟에서 얼굴을 찾지 못했습니다."); st.stop()
    if len(tgt_faces) < len(sources):
        st.warning(f"타겟 감지 {len(tgt_faces)}명 — 소스 {len(sources)}명보다 적습니다.")

    st.markdown("**타겟 검출 프리뷰**")
    st.image(bgr2rgb(draw_faces_preview(tgt_for_detect, tgt_faces)), use_container_width=True)

    # 매핑
    src_feats = [ s[2].normed_embedding for s in sources ]
    tgt_feats = [ f.normed_embedding for f in tgt_faces ]
    mapping, sims = map_sources_to_targets_gender_aware(
        [s[2] for s in sources], tgt_faces, src_feats, tgt_feats, gender_penalty=0.35
    )
    st.write("자동 매핑:", [(sources[i][0], j, round(sim,3)) for i,j,sim in mapping])

    st.markdown("**수동 매핑 (선택 시 적용)**")
    manual = []
    for i,(label,_,_,_) in enumerate(sources):
        manual.append(st.selectbox(f"{label} → 타겟 인덱스", options=list(range(len(tgt_faces))), index=mapping[i][1]))

    # 스왑 & 보정
    base = tgt_for_detect.copy()
    for i,(label, s_img, s_face, _) in enumerate(sources):
        t_idx = manual[i] if manual else mapping[i][1]
        t_face = tgt_faces[int(t_idx)]
        try:
            swapped_full = swapper.get(base.copy(), t_face, s_face, paste_back=True)
        except Exception as e:
            st.error(f"{label} 스왑 중 오류: {e}"); st.stop()

        x1,y1,x2,y2 = map(int, t_face.bbox)
        x1c,y1c,x2c,y2c = clamp_rect(x1,y1,x2,y2, base.shape[1], base.shape[0])
        patch_swapped = swapped_full[y1c:y2c, x1c:x2c]
        patch_target  = base[y1c:y2c, x1c:x2c]

        # 색동기화 + 기본 디테일
        if patch_swapped.size and patch_target.size:
            if keep_color:
                patch_swapped = reinhard_transfer(patch_swapped, patch_target)
            if detail_boost > 0.0:
                patch_swapped = unsharp_mask(patch_swapped, radius=1.1, amount=detail_boost)

            # NEW: 얼굴 초해상/복원
            if face_sr != "Off":
                patch_swapped = enhance_face_patch(patch_swapped, engine=engine, strength=face_sr)

        # 블렌딩
        if use_poisson:
            h, w = patch_swapped.shape[:2]
            mask = face_mask_from_wh(w, h, feather=0.15)
            center = ((x1c+x2c)//2, (y1c+y2c)//2)
            try:
                base = cv2.seamlessClone(patch_swapped, base, mask, center, cv2.NORMAL_CLONE)
            except Exception:
                base[y1c:y2c, x1c:x2c] = patch_swapped
        else:
            base[y1c:y2c, x1c:x2c] = patch_swapped

    out = base
    if not keep_prescaled and pre_scale > 1.0:
        out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
    if post_scale and post_scale > 1.0:
        out = cv2.resize(out, (int(out.shape[1]*post_scale), int(out.shape[0]*post_scale)), interpolation=cv2.INTER_CUBIC)

    st.success("완료! 아래 결과를 확인하세요.")
    st.image(bgr2rgb(out), use_container_width=True)
    ok = cv2.imencode(".png", out)[1].tobytes()
    st.download_button("결과 PNG 다운로드", data=ok, file_name="faceswapped.png", mime="image/png", use_container_width=True)

else:
    st.info("소스(A/B 또는 한 장 2명)와 타겟을 업로드 → 실행. 좌측에서 '얼굴 디테일 업스케일'을 'Medium/Strong'으로 올려보세요.")
