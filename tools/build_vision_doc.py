"""生成《视觉识别模块设计》PDF：先用 python-docx 写 docx，再用 LibreOffice Kit 转 PDF。"""
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
DOCX = os.path.abspath(os.path.join(OUT_DIR, "vision_design.docx"))

FONT = "微软雅黑"
MONO = "Consolas"


def set_run(run, size=10.5, bold=False, mono=False, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    name = MONO if mono else FONT
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text="", size=10.5, bold=False, mono=False, space_after=4,
         align=None, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align:
        p.alignment = align
    set_run(p.add_run(text), size=size, bold=bold, mono=mono, color=color)
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    set_run(p.add_run(text), size=16, bold=True, color=(0x1F, 0x3B, 0x66))
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    set_run(p.add_run(text), size=12.5, bold=True, color=(0x2E, 0x5A, 0x88))
    return p


def bullets(doc, items, size=10.5):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        set_run(p.add_run(it), size=size)


def code(doc, text, size=9):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        set_run(p.add_run(line if line else " "), size=size, mono=True,
                color=(0x33, 0x33, 0x33))
    para(doc, "", size=4, space_after=2)


def table(doc, header, rows, widths=None, size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        set_run(p.add_run(h), size=size, bold=True)
        shd = hdr[i]._tc.get_or_add_tcPr()
        el = shd.makeelement(qn("w:shd"), {qn("w:fill"): "E8EEF7"})
        shd.append(el)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            txt = str(v)
            mono = txt.startswith("`") and txt.endswith("`")
            set_run(p.add_run(txt.strip("`")), size=size, mono=mono)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    para(doc, "", size=4, space_after=4)


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = Document()

    # 页面与默认字体
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # ---------- 封面标题 ----------
    para(doc, "视觉识别模块设计", size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=6, color=(0x1F, 0x3B, 0x66))
    para(doc, "MuJoCo 双传送带取放项目 · vision/ 目录规划", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2, color=(0x66, 0x66, 0x66))
    para(doc, "方案：YOLOv8-seg + 单应标定 + 卡尔曼多目标跟踪",
         size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14,
         color=(0x66, 0x66, 0x66))

    h1(doc, "1  总体设计")
    para(doc, "目标：用俯视相机识别 A 带上数量不定的方块，输出世界坐标、速度和不确定度，"
              "替换强化学习环境中原本来自 MuJoCo 真值的观测量。")
    h2(doc, "1.1  数据流")
    code(doc, """
物理主线程 (100 Hz)
   MuJoCo 场景 ──渲染(birdeye)──► 图像 ndarray ──queue──► 视觉线程
                                                          │
视觉线程 (10~15 Hz)                                       │
   YOLOv8-seg 推理 ─► 实例掩码 ─► 亚像素质心 ─► 单应矩阵 ─► 世界坐标
                                                          │
   多目标卡尔曼跟踪 ─► 位置/速度/协方差 ─► 带时间戳快照 ──► 快照(VisionState)
                                                          │
控制线程 (100 Hz)  ◄── 快照 + 速度外推 ◄──────────────────┘
   外推到当前时刻 ─► obs[11:14] 目标位置, obs[19:22] σ
""")
    h2(doc, "1.2  设计原则")
    bullets(doc, [
        "渲染只能在主线程（GL 上下文），子线程只做推理与滤波；",
        "观测用估计值，成功判定仍用真值（评估时才知道真值，标准做法）；",
        "不确定度来自卡尔曼协方差与新息，而不是网络置信度；",
        "每个阶段都有可量化的验收指标（位置误差、数量正确率、端到端成功率）。",
    ])

    h1(doc, "2  文件架构")
    code(doc, """
scene_magnet/vision/
├── __init__.py           # 包入口                        （已有）
├── README.md             # 用法与流程                    （待写）
├── camera.py             # 相机与渲染、投影、单应矩阵      （已有）
├── calibrate.py          # 标定：像素↔带面坐标 + 残差验证  （已有）
├── scene.py              # 场景封装：方块摆放、光照随机化  （待写）
├── dataset.py            # YOLO-seg 数据集生成            （待写）
├── train_yolo.py         # 训练入口（ultralytics）        （待写）
├── detect.py             # 检测器：YOLOv8-seg / 颜色基线   （待写）
├── tracker.py            # 卡尔曼多目标跟踪               （待写）
├── vision_thread.py      # 视觉线程 + 快照发布            （待写）
├── bridge.py             # 与环境的接口                  （待写）
├── eval_vision.py        # 精度与鲁棒性评测              （待写）
└── calib.npz             # 标定产物（生成，不入库）

assets/models/vision_scene.xml   # train_scene + birdeye 俯视相机（已有）
""")
    h2(doc, "2.1  各文件职责")
    table(doc,
          ["文件", "职责", "输入 → 输出", "依赖"],
          [["camera.py", "离屏渲染；世界点↔像素", "MjData → 图像；点互转", "mujoco"],
           ["calibrate.py", "解析投影 + 红色质心交叉验证，产出标定", "— → H, H⁻¹", "camera"],
           ["scene.py", "加载 vision 场景；摆放方块；随机光照", "参数 → 场景状态", "camera"],
           ["dataset.py", "批量渲染 + 多边形标签 + data.yaml", "张数 → 数据集", "camera, scene"],
           ["train_yolo.py", "封装 ultralytics 训练", "data.yaml → best.pt", "ultralytics"],
           ["detect.py", "YOLO-seg 推理 → 掩码质心 + σ；颜色基线", "图像 → [Detection]", "cv2/ultralytics"],
           ["tracker.py", "恒速卡尔曼多目标跟踪、关联、遮挡续命", "[Detection] → [Track]", "numpy"],
           ["vision_thread.py", "固定频率取帧→检测→跟踪→发布快照", "— → VisionState", "上述全部"],
           ["bridge.py", "快照外推 → obs 的 11-14 / 19-22 维", "VisionState → (pos,vel,σ)", "tracker"],
           ["eval_vision.py", "位置误差/数量正确率/光照曲线/端到端", "— → 指标表", "全部"]],
          widths=[3.4, 6.6, 4.2, 3.0])

    h1(doc, "3  函数 / 类 / 方法清单")
    h2(doc, "3.1  camera.py（已有）")
    table(doc, ["名称", "类型", "说明"],
          [["load_model(path)", "函数", "加载 vision 场景"],
           ["Camera.__init__(w, h, scene)", "方法", "建 data / renderer / 相机 id"],
           ["Camera.render()", "方法", "update_scene + render → ndarray"],
           ["Camera.fit_homography(pixels, worlds)", "方法", "DLT 拟合像素→世界（≥4 点，纯 numpy）"],
           ["Camera.pixel_to_world / world_to_pixel", "方法", "坐标互转"],
           ["Camera.close()", "方法", "释放 renderer"],
           ["PLANE_Z = 0.375", "常量", "投影平面（方块中心高度）"],
           ["project_point(cam, xyz)", "函数", "待从 calibrate 迁入：解析投影"]],
          widths=[7.0, 2.2, 8.4], size=8.5)

    h2(doc, "3.2  calibrate.py（已有）")
    table(doc, ["名称", "类型", "说明"],
          [["project_point(cam, world_xyz)", "函数", "用 cam_pos / cam_mat0 / fovy 解析投影"],
           ["detect_red_centroid(img)", "函数", "红色掩码质心（交叉验证用）"],
           ["calibrate(w, h, verbose)", "函数", "12 网格点 → 投影 → 拟合 → 存 calib.npz"],
           ["load_calibration()", "函数", "读回 H, H⁻¹, meta"]],
          widths=[7.0, 2.2, 8.4], size=8.5)

    h2(doc, "3.3  scene.py / dataset.py / train_yolo.py（待写）")
    table(doc, ["名称", "说明"],
          [["VisionScene.reset()", "复位，隐藏所有方块"],
           ["VisionScene.place_cubes(list_xy)", "摆放方块（y 随机）"],
           ["VisionScene.set_lighting(level, rng)", "光照随机化：位置/色温/强度/高光/环境光"],
           ["VisionScene.cube_positions()", "当前真值（生成标签用）"],
           ["project_cube_polygon(cam, xyz)", "方块顶面 4 角 → 像素多边形"],
           ["render_sample(scene, cam, level)", "渲染一帧 + 标签"],
           ["build_dataset(n, out, levels)", "批量生成，train/val = 8:2"],
           ["write_data_yaml(...)", "path / train / val / names: [cube]"],
           ["train(data_yaml, weights, epochs, imgsz)", "封装 YOLO(...).train(...)"],
           ["main()", "CLI 参数解析与训练入口"]],
          widths=[8.6, 9.0], size=8.5)

    h2(doc, "3.4  detect.py（待写）")
    table(doc, ["名称", "说明"],
          [["@dataclass Detection", "xy_world, conf, area_px, sigma_mm, mask"],
           ["SegDetector.__init__(weights, calib)", "加载 YOLO-seg 与标定"],
           ["SegDetector.detect(img)", "results[0].masks.data → 质心 → 单应 → 世界坐标"],
           ["ColorDetector.detect(img)", "HSV + 连通域基线（训练前即可跑通全链路）"],
           ["mask_centroid(mask)", "亚像素质心（面积加权）"],
           ["mask_sigma(mask)", "由面积/圆度给 σ"]],
          widths=[8.6, 9.0], size=8.5)

    h2(doc, "3.5  tracker.py（待写）")
    table(doc, ["名称", "说明"],
          [["class KalmanTrack", "单目标：x=[px,py,vx,vy]，P 为 4×4"],
           ["KalmanTrack.predict(dt)", "x = Fx, P = FPFᵀ + Q"],
           ["KalmanTrack.update(z, R)", "K = PHᵀS⁻¹，更新 x、P，记录新息"],
           ["KalmanTrack.pos / vel / sigma / miss", "对外接口"],
           ["class MultiTracker", "多目标管理"],
           ["MultiTracker.step(dets, t)", "关联（最近邻 + 2 cm 门限）→ 更新/新建/续命"],
           ["MultiTracker.predicted(dt)", "预测 dt 之后的位置（遮挡与延迟补偿）"],
           ["MultiTracker.active()", "当前有效目标（按 x 排序）"]],
          widths=[8.6, 9.0], size=8.5)

    h2(doc, "3.6  vision_thread.py / bridge.py（待写）")
    table(doc, ["名称", "说明"],
          [["@dataclass VisionState", "t_capture, t_processed, objects[]"],
           ["VisionThread.run()", "主循环：取帧 → 检测 → 跟踪 → 发布快照"],
           ["VisionThread.submit_frame(img, t)", "主线程投递渲染帧（queue）"],
           ["VisionThread.snapshot()", "线程安全地取最新快照"],
           ["VisionBridge.query(now)", "快照 + 外推 → (pos, vel, σ)"],
           ["VisionBridge.use_vision", "开启后 env 用估计值替换 box_pos / box_vel / pos_std"]],
          widths=[8.6, 9.0], size=8.5)

    h1(doc, "4  知识清单")
    h2(doc, "4.1  卡尔曼滤波（核心）")
    para(doc, "状态与模型（恒速 CV 模型）：")
    code(doc, """
x = [px, py, vx, vy]ᵀ
F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
H = [[1,0,0,0],[0,1,0,0]]                 # 只观测位置
Q ≈ σa² · [[dt⁴/4,0,dt³/2,0], …]          # 过程噪声（加速度不确定度）
R = diag(σpx², σpy²)                      # 检测噪声（由标定/统计得到）
""")
    para(doc, "两步递推：")
    code(doc, """
预测  x⁻ = F x,            P⁻ = F P Fᵀ + Q
更新  S  = H P⁻ Hᵀ + R
      K  = P⁻ Hᵀ S⁻¹                 ← 卡尔曼增益（信任模型还是信任观测）
      x  = x⁻ + K (z − H x⁻)         ← 新息 innovation
      P  = (I − K H) P⁻
""")
    table(doc, ["概念", "含义", "本项目用途"],
          [["P（协方差）", "估计的不确定度", "直接填 obs 的 σx, σy, σvx, σvy"],
           ["新息 z − Hx⁻", "观测与预测之差", "检测质量监控、异常值剔除（NIS 检验）"],
           ["增益 K", "模型与观测的加权", "检测噪声大 → K 小 → 更信模型"],
           ["Q / R 调参", "过程/观测噪声", "R 由检测误差统计，Q 由加速度先验"]],
          widths=[3.6, 6.0, 8.0], size=8.5)
    para(doc, "遮挡处理（正对应本项目问题）：")
    code(doc, """
检测到    → predict + update       （P 变小，σ 变小）
未检测到  → 只 predict，miss += 1  （P 变大，σ 变大，位置沿速度外推）
miss 超阈值（如 5 帧 / 0.5 s）→ 删除 track
""")
    bullets(doc, [
        "初始化：单帧无法定速度，用前两帧位置差给 v，P_v 初始给大一些；",
        "多目标关联：距离门限（约 2 cm）+ 最近邻；按 x 排序做二次校验，防止身份互换；",
        "外推：x(t+Δ) = px + vx·Δ。带速恒定 ⇒ 恒速模型无模型误差，这是本项目的天然优势；",
        "局限：目标机动需 CA 模型或自适应 Q；非线性观测需 EKF/UKF（本项目在像素域滤波 + 单应变换即可）。",
    ])

    h2(doc, "4.2  相机与几何")
    table(doc, ["知识点", "要点"],
          [["针孔模型", "u = f·X/Z + cx，v = f·Y/Z + cy"],
           ["内参", "f 由 fovy 与分辨率确定，cx/cy 通常为图像中心"],
           ["外参", "R, t（MuJoCo：cam_pos、cam_mat0）"],
           ["相机朝向", "MuJoCo 相机沿局部 −z 看，图像 y 向下"],
           ["单应矩阵", "平面→平面投影，8 自由度，≥4 点，DLT 用 SVD 求解"],
           ["往返误差", "用世界→像素→世界检验，应 < 1 mm"],
           ["亚像素质心", "掩码像素坐标的面积加权平均"],
           ["视差误差", "投影假设 z 恒定；方块高 5 cm，取中心 0.375 可压到约 1 px"]],
          widths=[3.4, 14.2], size=8.5)

    h2(doc, "4.3  检测与精度")
    table(doc, ["知识点", "要点"],
          [["掩码 vs 框", "掩码质心亚像素；框中心有整数像素量化，差 3~8 mm"],
           ["置信度", "YOLO 的 conf 是分类/定位置信，不能直接当位置 σ"],
           ["σ 的正确来源", "① 掩码面积/形状 ② 卡尔曼新息 ③ 多帧拟合残差"],
           ["位置指标", "位置误差中位数与 95 分位（mm）、数量正确率、漏检/误检率"],
           ["训练指标", "mask mAP50 / IoU 只说明像不像，不说明准不准"],
           ["评测方式", "固定测试集 + 光照档位扫描 + 遮挡模拟"]],
          widths=[3.4, 14.2], size=8.5)

    h2(doc, "4.4  光照鲁棒性")
    table(doc, ["手段", "说明"],
          [["域随机化", "光源位置/色温/强度/高光、材质、曝光、噪声、运动模糊"],
           ["评测协议", "光照档位 0~3（无/弱/中/强），画成功率-档位曲线"],
           ["传统方法对策", "HSV/色度归一化、背景差分、自适应阈值（作为基线对比）"]],
          widths=[3.4, 14.2], size=8.5)

    h2(doc, "4.5  线程与实时性")
    table(doc, ["知识点", "要点"],
          [["GL 上下文", "mujoco.Renderer 只能主线程调用，子线程做推理"],
           ["数据通路", "queue.Queue 传 ndarray（渲染线程 → 视觉线程）"],
           ["时间戳", "快照必须带 t_capture，控制端按 now − t_capture 外推"],
           ["频率选择", "视觉 10~15 Hz、物理 100 Hz，延迟 60~100 ms 用速度补偿"],
           ["线程安全", "快照用不可变 dataclass 或加锁，避免读写撕裂"]],
          widths=[3.4, 14.2], size=8.5)

    h2(doc, "4.6  验收标准")
    table(doc, ["阶段", "指标", "目标"],
          [["标定", "像素往返误差", "< 1 px"],
           ["检测", "位置误差中位 / 95 分位", "< 2 mm / < 5 mm"],
           ["数量", "检测数量 == 真实数量", "> 98%"],
           ["跟踪", "遮挡 0.3 s 后位置误差", "< 10 mm"],
           ["光照", "档位 0~3 的位置误差", "不劣化 3 倍以上"],
           ["端到端", "接力成功率（视觉 vs 真值）", "下降 < 10 个百分点"]],
          widths=[3.0, 8.0, 6.6], size=8.5)

    h1(doc, "5  实施顺序与风险")
    table(doc, ["步骤", "内容", "产出"],
          [["1", "收尾标定：验证解析投影与检测质心一致（< 1 px）", "calib.npz + 误差数字"],
           ["2", "场景与数据：光照随机化 + 多边形标签 + data.yaml", "合成数据集"],
           ["3", "训练 YOLOv8-seg（n-seg 起步）", "best.pt + mAP"],
           ["4", "检测封装：掩码质心 + 单应 + σ", "位置误差指标"],
           ["5", "卡尔曼跟踪 + 视觉线程 + 快照", "遮挡实验数据"],
           ["6", "接入环境：use_vision 开关，替换 obs 的 11-14 / 19-22 维", "端到端成功率"]],
          widths=[1.6, 11.0, 5.0], size=8.5)
    para(doc, "主要风险与对策：")
    bullets(doc, [
        "标注漂移：标签用顶面四角投影多边形，验证时抽查 20 张叠加显示；",
        "掩码质心偏差：与真值对比统计系统误差，必要时加固定偏置；",
        "遮挡导致 ID 互换：按 x 排序 + 距离门限双重约束；",
        "延迟未补偿：所有位置都带时间戳，控制端统一外推；",
        "光照过强导致漏检：数据里加同等强度的随机化，评测时给档位曲线。",
    ])
    para(doc, "", size=6)
    para(doc, "注：本文件由脚本自动生成，源文件 docs/vision_design.docx。",
         size=8.5, color=(0x88, 0x88, 0x88))

    doc.save(DOCX)
    print("docx 已生成:", DOCX)


if __name__ == "__main__":
    build()
