# app.py — Miromi Wedding Face Swap (2 faces, gender-aware, natural blending)

import streamlit as st
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

st.set_page_config(page_title="Miromi Wedding Face Swap (2 faces)", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def read_image(file) -> np.ndarray:
    if hasattr(file, "read"):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim > 1: a = a.flatten()
    if b.ndim > 1: b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

def draw_faces_preview(img_bgr, faces, color=(0,255,0)):
    vis = img_bgr.copy()
    for idx, f in enumerate(faces):
        x1,y1,x2,y2 = map(int, f.bbox)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"#{idx}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis

# -----------------------------
# Gender-aware matching helpers
# -----------------------------
def get_sex(face):
    # InsightFace 버전에 따라 sex 또는 gender 속성을 제공
    val = getattr(face, "sex", None)
    if val is None:
        val = getattr(face, "gender", None)
    # 표준화: 남성=M, 여성=F, 모름=None
    if isinstance(val, (int, float)):
        if val == 1: return "M"   # 일부 버전은 1=male
        if val == 0: return "F"   # 0=female
    if isinstance(val, str):
        v = val.lower()
        if "m" in v: return "M"
        if "f" in v: return "F"
    return None

def gender_label(x):
    return {"M":"남","F":"여", None:"불명"}[x]

def map_sources_to_targets_gender_aware(src_faces, tgt_faces, src_feats, tgt_feats, gender_penalty=0.35):
    """
    2명 기준 최적 매핑: 성별 일치 우선 + 코사인 유사도.
    cost = -similarity + penalty(성별 다르면)
    """
    m = len(src_faces); n = len(tgt_faces)
    assert m in (1,2)
    src_g = [get_sex(f) for f in src_faces]
    tgt_g = [get_sex(f) for f in tgt_faces]

    sims = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            sims[i, j] = cosine_sim(src_feats[i], tgt_feats[j])

    def cost_of(pairing):
        c = 0.0
        for (i, j) in pairing:
            c += -sims[i, j]
            if src_g[i] is not None and tgt_g[j] is not None and src_g[i] != tgt_g[j]:
                c += gender_penalty
        return c

    if m == 1:
        best_j, best_c = None, 1e9
        best_sim = None
        for j in range(n):
            cc = cost_of([(0, j)])
            if cc < best_c:
                best_c = cc; best_j = j; best_sim = sims[0, j]
        return [(0, best_j, float(best_sim))], sims

    # m == 2
    import itertools
    best_pair, best_c, best_sims = None, 1e9, None
    for j0, j1 in itertools.permutations(range(n), 2):
        pairing = [(0, j0), (1, j1)]
        cc = cost_of(pairing)
        if cc < best_c:
            best_c = cc; best_pair = pairing
            best_sims = (sims[0, j0], sims[1, j1])
    mapping = [(best_pair[0][0], best_pair[0][1], float(best_sims[0])),
               (best_pair[1][0], best_pair[1][1], float(best_sims[1]))]
    return mapping, sims

# -----------------------------
# Color/Boundary harmonization
# -----------------------------
def reinhard_transfer(src_bgr, ref_bgr):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        s_mean, s_std = src[:,:,ch].mean(), src[:,:,ch].std() + 1e-6
        r_mean, r_std = ref[:,:,ch].mean(), ref[:,:,ch].std() + 1e-6
        src[:,:,ch] = (src[:,:,ch] - s_mean) * (r_std / s_std) + r_mean
    out = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out

def face_mask_from_bbox_wh(width, height, feather=0.15):
    mask = np.zeros((height, width), np.uint8)
    cx, cy = width//2, height//2
    rx, ry = int(width*0.5), int(height*0.5)
    cv2.ellipse(mask, (cx,cy), (rx,ry), 0, 0, 360, 255, -1)
    if feather>0:
        k = int(max(3, (width+height)*0.5*feather))
        if k % 2 == 0: k += 1
        mask = cv2.GaussianBlur(mask, (k,k), 0)
    return mask

# -----------------------------
# Model loader (GPU toggle-safe)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models(use_gpu=False, det_size=(640,640)):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    # Face detector/recognition
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=(0 if use_gpu else -1), det_size=det_size)
    # Face swapper
    swapper = get_model('inswapper_128.onnx', download=True, download_zip=True, providers=providers)
    return app, swapper

# -----------------------------
# UI
# -----------------------------
st.title("Miromi Wedding Face Swap")
st.caption("Developed for event use • 2인 얼굴 스왑 • ID 일관성 유지 • InsightFace")

with st.sidebar:
    st.header("세팅")
    gpu = st.checkbox("GPU 사용 (CUDA)", value=False)   # 기본값 False(안전)
    det = st.select_slider("검출 해상도", [320, 480, 640, 800, 960], value=800)
    app, swapper = load_models(use_gpu=gpu, det_size=(det, det))

st.subheader("1) 소스 얼굴(사용자) 업로드 — 최대 2명")
c1, c2 = st.columns(2)
src_files = []
with c1:
    f1 = st.file_uploader("소스 A", type=["jpg","jpeg","png"], key="srcA")
    if f1: src_files.append(("A", f1))
with c2:
    f2 = st.file_uploader("소스 B", type=["jpg","jpeg","png"], key="srcB")
    if f2: src_files.append(("B", f2))

st.subheader("2) 타겟 웨딩 사진 업로드 (두 얼굴이 보이는 사진 권장)")
tfile = st.file_uploader("타겟", type=["jpg","jpeg","png"], key="target")

st.subheader("옵션")
colA, colB, colC = st.columns(3)
with colA:
    keep_color = st.checkbox("피부톤/컬러 최대한 유지", value=True)
with colB:
    use_poisson = st.checkbox("경계 블렌딩(Poisson)", value=True)
with colC:
    sharpen = st.checkbox("약한 샤픈", value=True)

run = st.button("얼굴 스왑 실행", type="primary", use_container_width=True)

# -----------------------------
# Main logic
# -----------------------------
if run:
    if len(src_files) == 0 or tfile is None:
        st.error("소스 얼굴 1~2장과 타겟 사진을 모두 업로드해 주세요.")
        st.stop()

    # Read sources & detect
    sources = []
    for label, f in src_files:
        try:
            img = read_image(f)
        except Exception as e:
            st.error(f"{label} 이미지를 읽을 수 없습니다: {e}")
            st.stop()
        sfaces = app.get(img)
        if len(sfaces) == 0:
            st.error(f"{label}에서 얼굴을 찾지 못했습니다.")
            st.stop()
        # 가장 큰 얼굴 선택
        areas = [ (sf.bbox[2]-sf.bbox[0])*(sf.bbox[3]-sf.bbox[1]) for sf in sfaces ]
        s_pick = sfaces[int(np.argmax(areas))]
        sources.append((label, img, s_pick))

    # Read target & detect
    tgt_img = read_image(tfile)
    tgt_faces = app.get(tgt_img)
    if len(tgt_faces) == 0:
        st.error("타겟에서 얼굴을 찾지 못했습니다.")
        st.stop()
    if len(tgt_faces) < len(sources):
        st.warning(f"타겟에서 {len(tgt_faces)}명만 감지됨 — 업로드한 소스 수({len(sources)})보다 적습니다.")

    # Preview detections
    st.markdown("**검출 결과 프리뷰** (타겟 얼굴 인덱스 확인)")
    prev = draw_faces_preview(tgt_img, tgt_faces)
    st.image(bgr2rgb(prev), use_column_width=True)

    # Build features & gender-aware mapping
    src_feats = [ s[2].normed_embedding for s in sources ]
    tgt_feats = [ f.normed_embedding for f in tgt_faces ]

    mapping, sims = map_sources_to_targets_gender_aware(
        [s[2] for s in sources], tgt_faces, src_feats, tgt_feats, gender_penalty=0.35
    )

    st.write("소스 성별:", [ (sources[i][0], gender_label(get_sex(sources[i][2]))) for i in range(len(sources)) ])
    st.write("타겟 성별:", [ (idx, gender_label(get_sex(tf))) for idx, tf in enumerate(tgt_faces) ])
    st.write("자동 매핑(유사도):", [(sources[i][0], j, round(sim,3)) for i, j, sim in mapping])

    # Manual override
    st.markdown("**수동 매핑 (선택 시 수동 적용)**")
    manual = []
    for i,(label,_,_) in enumerate(sources):
        choice = st.selectbox(f"{label} → 타겟 얼굴 인덱스", options=list(range(len(tgt_faces))), index=mapping[i][1])
        manual.append(choice)

    # Swapping + harmonization
    base = tgt_img.copy()  # 최종 결과 누적
    for i,(label, s_img, s_face) in enumerate(sources):
        t_idx = manual[i] if manual else mapping[i][1]
        t_face = tgt_faces[int(t_idx)]
        try:
            # 전체 이미지 기준 스왑(패치 생성용)
            swapped_full = swapper.get(base.copy(), t_face, s_face, paste_back=True)
        except Exception as e:
            st.error(f"{label} 스왑 중 오류: {e}")
            st.stop()

        # 타겟 bbox 패치 추출
        x1,y1,x2,y2 = map(int, t_face.bbox)
        x1c, y1c = max(0,x1), max(0,y1)
        x2c, y2c = min(base.shape[1],x2), min(base.shape[0],y2)
        if x2c<=x1c or y2c<=y1c:
            continue

        patch_swapped = swapped_full[y1c:y2c, x1c:x2c]
        patch_target  = base[y1c:y2c, x1c:x2c]

        # 색 맞춤 (Reinhard)
        if keep_color and patch_swapped.size>0 and patch_target.size>0:
            patch_swapped = reinhard_transfer(patch_swapped, patch_target)

        # 경계 블렌딩(Poisson)
        if use_poisson:
            h, w = patch_swapped.shape[:2]
            mask_small = face_mask_from_bbox_wh(w, h, feather=0.15)
            try:
                base = cv2.seamlessClone(
                    patch_swapped, base, mask_small, ((x1c+x2c)//2, (y1c+y2c)//2),
                    cv2.NORMAL_CLONE
                )
            except Exception:
                # 드문 실패 시: 패치 덮어쓰기 fallback
                base[y1c:y2c, x1c:x2c] = patch_swapped
        else:
            base[y1c:y2c, x1c:x2c] = patch_swapped

    out = base

    # Sharpen
    if sharpen:
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, k)

    st.success("완료! 아래 결과를 확인해 주세요.")
    st.image(bgr2rgb(out), use_column_width=True)

    # Download
    ok = cv2.imencode(".png", out)[1].tobytes()
    st.download_button("결과 PNG 다운로드", data=ok, file_name="faceswapped.png", mime="image/png", use_container_width=True)

else:
    st.info("좌측에서 GPU 옵션 확인 → 소스 A/B와 타겟 업로드 → '얼굴 스왑 실행'. 성별 고려 자동매핑 + 수동 인덱스로 뒤바뀜 방지 가능.")
