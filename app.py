import streamlit as st
import sympy as sp
import numpy as np
import control as ctrl
import google.generativeai as genai
from PIL import Image

st.set_page_config(page_title="進階貝祖恆等式計算器", layout="wide")

st.title("🎛️ 互質因式分解與擴展貝祖恆等式驗證 (含圖片與 G(s) 輸入)")

# --- 側邊欄：輸入模式選擇 ---
st.sidebar.header("系統輸入設定")
input_mode = st.sidebar.radio(
    "選擇輸入模式", 
    ["1. 輸入狀態空間 (A, B, C, D)", "2. 文字輸入轉移函數 G(s)", "3. 圖片辨識轉移函數"]
)

str_poles = st.sidebar.text_input("期望閉迴路極點 (配置 F 與 L，用逗號隔開)", "-2, -3")
desired_poles = [float(p.strip()) for p in str_poles.split(',') if p.strip()]

# 全局變數存放轉換出來的 ABCD
A, B, C, D = None, None, None, None
s = sp.Symbol('s')

# ==========================================
# 模式 1: 傳統 ABCD 矩陣輸入
# ==========================================
if input_mode == "1. 輸入狀態空間 (A, B, C, D)":
    st.subheader("📝 模式 1：直接輸入 ABCD 矩陣")
    col1, col2 = st.columns(2)
    with col1:
        str_A = st.text_area("矩陣 A (列用分號或換行隔開)", "0, 1\n 2, -1", height=100)
        str_C = st.text_area("矩陣 C", "1, 0", height=60)
    with col2:
        str_B = st.text_area("矩陣 B", "0\n 1", height=100)
        str_D = st.text_area("矩陣 D", "0", height=60)

    def parse_matrix(s_str):
        rows = [r.strip() for r in s_str.replace(';', '\n').split('\n') if r.strip()]
        return np.array([[float(x.strip()) for x in r.replace(',', ' ').split() if x.strip()] for r in rows])

    if st.button("開始計算"):
        A, B, C, D = parse_matrix(str_A), parse_matrix(str_B), parse_matrix(str_C), parse_matrix(str_D)

# ==========================================
# 模式 2: 文字輸入 G(s)
# ==========================================
elif input_mode == "2. 文字輸入轉移函數 G(s)":
    st.subheader("📝 模式 2：輸入轉移函數 G(s) (目前支援 SISO 系統)")
    g_str = st.text_input("輸入以 s 為變數的轉移函數 (例如: 1/(s^2 + 2*s - 3))", "1/(s - 1)")
    
    if st.button("解析 G(s) 並計算"):
        try:
            # 1. 將字串轉為 SymPy 符號式並提取分子分母
            expr = sp.sympify(g_str)
            num, den = sp.fraction(sp.cancel(expr))
            
            # 2. 提取係數
            num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
            den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]
            
            # 3. 透過 control 套件轉為狀態空間 ABCD
            sys_tf = ctrl.TransferFunction(num_coeffs, den_coeffs)
            sys_ss = ctrl.tf2ss(sys_tf)
            A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
            st.success(f"成功將 $G(s)$ 轉換為狀態空間模型 (維度: {A.shape[0]}階)")
        except Exception as e:
            st.error(f"解析失敗，請確認數學式語法 (例如乘號要用 *): {e}")

# ==========================================
# 模式 3: 圖片上傳辨識 G(s)
# ==========================================
elif input_mode == "3. 圖片辨識轉移函數":
    st.subheader("📸 模式 3：上傳圖片自動辨識 G(s)")
    
    # 🌟 在網頁畫面上提供 API Key 輸入框（密碼隱藏格式）
    user_api_key = st.text_input(
        "請輸入您的 Gemini API Key", 
        type="password", 
        placeholder="在此貼上 AIzaSy 開頭的 API Key..."
    )
    
    uploaded_file = st.file_uploader("上傳包含 G(s) 的圖片 (支援 JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="您上傳的圖片", use_container_width=True)
        
        if st.button("開始 AI 辨識並計算"):
            if not user_api_key.strip():
                st.error("❌ 請先在上方輸入您的 Gemini API Key 才能啟用辨識！")
            else:
                with st.spinner("AI 正在辨識圖片中的數學式..."):
                    try:
                        # 呼叫使用者在網頁輸入的 API Key
                        genai.configure(api_key=user_api_key.strip())
                        # 1. 更新為系統要求的新版模型
                        model = genai.GenerativeModel('gemini-3.8-flash')

                        # 2. 精準 prompt，讓 AI 只輸出純公式
                        prompt = "這是一張控制系統的轉移函數圖片。請提取其中的數學表達式，只輸出單行式子（變數為 s，例如 100/s 或 1/(s^2+2*s+1)），不要包含 'G(s)='，不要包含任何 markdown 或其他文字。"
                        response = model.generate_content([prompt, image])

                        # 3. 防呆過濾：去除空白與可能多帶的 G(s)=
                        recognized_str = response.text.strip().replace('`', '').replace(' ', '')
                        if "=" in recognized_str:
                            recognized_str = recognized_str.split("=")[-1]

                        # 避免 AI 還是吐出 "G(s)="，自動去掉等號左邊
                        if "=" in recognized_str:
                            recognized_str = recognized_str.split("=")[-1]
                        
                        st.success(f"AI 辨識結果: {recognized_str}")
                        
                        # 將辨識出的字串送入 SymPy 與 Control 處理
                        expr = sp.sympify(recognized_str)
                        num, den = sp.fraction(sp.cancel(expr))
                        num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
                        den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]
                        
                        sys_tf = ctrl.TransferFunction(num_coeffs, den_coeffs)
                        sys_ss = ctrl.tf2ss(sys_tf)
                        A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
                    except Exception as e:
                        st.error(f"辨識或解析失敗，請確認 API Key 是否有效，或嘗試更清晰的圖片: {e}")

# ==========================================
# 核心運算引擎 (100% 吻合講義第 25 頁 Two-Port 定義)
# ==========================================
if A is not None and B is not None and C is not None and D is not None:
    try:
        st.markdown("---")
        st.subheader("🚀 互質因式分解與貝祖恆等式運算結果 (講義標準型)")
        
        n = A.shape[0]
        p, m = C.shape[0], B.shape[1]
        
        # 1. 極點配置
        K_f = ctrl.place(A, B, desired_poles)
        F = -K_f
        K_l = ctrl.place(A.T, C.T, desired_poles)
        L = -K_l.T

        # 符號狀態空間轉移函數
        def ss2tf_sym(A_m, B_m, C_m, D_m):
            A_sp, B_sp, C_sp, D_sp = sp.Matrix(A_m), sp.Matrix(B_m), sp.Matrix(C_m), sp.Matrix(D_m)
            resolvent = (s * sp.eye(A_m.shape[0]) - A_sp).inv()
            return sp.simplify(C_sp * resolvent * B_sp + D_sp)

        with st.spinner("正在進行符號運算，請稍候..."):
            I_m, I_p = np.eye(m), np.eye(p)
            
            # --- 依據講義第 25 頁右側區塊: [M, Y; N, X] ---
            # A_cl = A + B@F
            # B_cl = [B, L]
            # C_cl = [F; C + D@F]
            # D_cl = [I, 0; D, -I]
            M_s = ss2tf_sym(A + B @ F, B, F, I_m)
            Y_s = ss2tf_sym(A + B @ F, L, F, np.zeros((m, p)))
            N_s = ss2tf_sym(A + B @ F, B, C + D @ F, D)
            X_s = ss2tf_sym(A + B @ F, L, C + D @ F, -I_p)

            # --- 依據講義第 25 頁左側區塊: [X~, Y~; N~, M~] ---
            # A_cl = A + L@C
            # B_cl = [B + L@D, -L]
            # C_cl = [-F; C]
            # D_cl = [I, 0; D, -I]
            X_tilde_s = ss2tf_sym(A + L @ C, B + L @ D, -F, I_m)
            Y_tilde_s = ss2tf_sym(A + L @ C, -L, -F, np.zeros((m, p)))
            N_tilde_s = ss2tf_sym(A + L @ C, B + L @ D, C, D)
            M_tilde_s = ss2tf_sym(A + L @ C, -L, C, -I_p)

        # 顯示 8 個因子
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("### 📌 右側矩陣因子")
            st.write("**$M(s):$**"); st.latex(sp.latex(M_s))
            st.write("**$N(s):$**"); st.latex(sp.latex(N_s))
            st.write("**$X(s):$**"); st.latex(sp.latex(X_s))
            st.write("**$Y(s):$**"); st.latex(sp.latex(Y_s))
            
        with col_res2:
            st.write("### 📌 左側矩陣因子 (帶波浪 / 講義型)")
            st.write("**$\\tilde{M}(s):$**"); st.latex(sp.latex(M_tilde_s))
            st.write("**$\\tilde{N}(s):$**"); st.latex(sp.latex(N_tilde_s))
            st.write("**$\\tilde{X}(s):$**"); st.latex(sp.latex(X_tilde_s))
            st.write("**$\\tilde{Y}(s):$**"); st.latex(sp.latex(Y_tilde_s))
            
        st.write("### 🎯 擴展貝祖恆等式驗證 (講義第 25 頁矩陣乘法)")
        st.markdown(r"""
        $$\begin{bmatrix} \tilde{X}(s) & \tilde{Y}(s) \\ \tilde{N}(s) & \tilde{M}(s) \end{bmatrix} \begin{bmatrix} M(s) & Y(s) \\ N(s) & X(s) \end{bmatrix} = \begin{bmatrix} I & 0 \\ 0 & I \end{bmatrix}$$
        """)
        
        # 定義一個專門清理浮點數雜訊與自動約分的函式
        def clean_poly_expr(expr):
            try:
                # 1. 濾除小於 1e-5 的浮點數雜訊（把 10^-12 歸零）
                expr_rational = sp.nsimplify(expr, tolerance=1e-5, rational=True)
                # 2. 自動進行分子分母公因式約分 (把 P(s)/P(s) 約分成 1)
                return sp.cancel(expr_rational)
            except Exception:
                return sp.simplify(expr)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            eq1 = clean_poly_expr(X_tilde_s * M_s + Y_tilde_s * N_s)
            st.markdown(r"**1. 左上項：** $\tilde{X}M + \tilde{Y}N =$")
            st.latex(sp.latex(eq1))
            
            eq3 = clean_poly_expr(N_tilde_s * M_s + M_tilde_s * N_s)
            st.markdown(r"**3. 左下項：** $\tilde{N}M + \tilde{M}N =$")
            st.latex(sp.latex(eq3))
            
        with col_v2:
            eq2 = clean_poly_expr(X_tilde_s * Y_s + Y_tilde_s * X_s)
            st.markdown(r"**2. 右上項：** $\tilde{X}Y + \tilde{Y}X =$")
            st.latex(sp.latex(eq2))
            
            eq4 = clean_poly_expr(N_tilde_s * Y_s + M_tilde_s * X_s)
            st.markdown(r"**4. 右下項：** $\tilde{N}Y + \tilde{M}X =$")
            st.latex(sp.latex(eq4))

    except Exception as e:
        st.error(f"系統階數與期望極點數量不匹配或發生錯誤：{e}")