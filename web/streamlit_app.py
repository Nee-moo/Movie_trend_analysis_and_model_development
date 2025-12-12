import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np 
import joblib

st.set_page_config(page_title="Movie Dashboard", layout="wide")
df = pd.read_csv("../data/processed/cleaned_data.csv")

try:
    model = joblib.load('random_forest_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("⚠️ Không tìm thấy file mô hình!")
    st.stop()

st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
)

# --- 2. TẢI MÔ HÌNH ---
@st.cache_resource
def load_model_resources():
    try:
        # Load file model và danh sách cột chuẩn
        model = joblib.load('random_forest_model.joblib')
        cols = joblib.load('model_columns.joblib')
        return model, cols
    except FileNotFoundError:
        return None, None

model, model_columns = load_model_resources()

if model is None:
    st.error("⚠️ LỖI: Không tìm thấy file mô hình!")
    st.info("👉 Hãy chạy file `model/random_forest.py` trước để tạo file .joblib")
    st.stop()

# --- 3. GIAO DIỆN NHẬP LIỆU ---
st.title("🎬 Dự Đoán Doanh Thu Phim")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Năm sản xuất (Year)", min_value=1900, max_value=2030, value=2024)
    rating = st.slider("Điểm đánh giá (Rating)", 0.0, 10.0, 7.0, step=0.1)

with col2:
    vote_count = st.number_input("Lượt bình chọn (Vote Count)", min_value=0, value=5000, step=100)
    
    # Chỉ tính Social_Buzz (vì có thể bạn có dùng), bỏ qua Vote_Squared và Movie_Age
    social_buzz = rating * vote_count

# 1. Lấy danh sách Thể loại
# Máy tính tự tìm các cột bắt đầu bằng "Genre_" và cắt bỏ tiền tố đi để hiển thị cho đẹp
all_genres = [col.replace("Genre_", "") for col in model_columns if col.startswith("Genre_")]
all_genres.sort() # Sắp xếp A-Z

# 2. Lấy danh sách Quốc gia
all_countries = [col.replace("Country_", "") for col in model_columns if col.startswith("Country_")]
all_countries.sort()

selected_genres = st.multiselect("Chọn Thể loại:", all_genres, default=['Action'])
selected_countries = st.multiselect("Chọn Quốc gia:", all_countries, default=['United States of America'])

# --- 4. DỰ ĐOÁN ---
st.markdown("---")
if st.button("🚀 Dự đoán Doanh thu", type="primary"):
    
    # BƯỚC 1: Tạo bảng dữ liệu rỗng chuẩn form mẫu (toàn số 0)
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 
    
    # BƯỚC 2: Điền các chỉ số cơ bản
    # Code tự động kiểm tra: Nếu mô hình CÓ cột đó thì mới điền, KHÔNG thì thôi (tránh lỗi)
    if 'Year' in input_data.columns: input_data['Year'] = year
    if 'Rating' in input_data.columns: input_data['Rating'] = rating
    if 'Vote_Count' in input_data.columns: input_data['Vote_Count'] = vote_count
    
    # Điền Social Buzz (nếu mô hình của bạn có dùng)
    if 'Social_Buzz' in input_data.columns: input_data['Social_Buzz'] = social_buzz

    # BƯỚC 3: Điền One-Hot Encoding (Thể loại & Quốc gia)
    # Tìm cột tên "Genre_Action", nếu có trong mô hình thì bật lên 1
    for g in selected_genres:
        col_name = f"Genre_{g}"  # Tự động ghép lại tiền tố để tìm cột
        if col_name in input_data.columns:
            input_data[col_name] = 1
            
    for c in selected_countries:
        col_name = f"Country_{c}"
        if col_name in input_data.columns:
            input_data[col_name] = 1
            
    # BƯỚC 4: Dự đoán và Đổi tiền
    try:
        # Dự đoán ra Log
        prediction_log = model.predict(input_data)
        
        # Đổi Log về Tiền thật
        prediction_real = np.expm1(prediction_log)[0]
        
        # Hiển thị kết quả
        st.success(f"💰 Doanh thu dự đoán: **${prediction_real:,.0f}**")
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")
        
st.markdown("---")

def get_unique_items(df, column_name):
    all_items = set()
    for item_str in df[column_name].dropna():
        items_list = [item.strip() for item in str(item_str).split(',') if item.strip()]
        all_items.update(items_list)
    return sorted(list(all_items))

unique_genres = get_unique_items(df, "Genres")
unique_countries = get_unique_items(df, "Production_Countries")

filtered_df = df.copy()
    
st.header("🏆 Xếp hạng Phim (Hệ số 0.0 - 1.0)")
st.markdown("Chọn trọng số theo thang thập phân. Tổng luôn bằng **1.0**.")

col_control, col_display = st.columns([1, 1])

with col_control:
    st.subheader("1. Điều chỉnh trọng số")
    
    # --- THANH 1: RATING (0.0 đến 1.0) ---
    w_rating = st.slider(
       "⭐ 1. Điểm đánh giá (Rating)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,    # Mặc định 0.5
        step=0.1,    # Bước nhảy 0.1
        key="slider_rating"
    )
        
    # --- THANH 2: DOANH THU ---
    # Tính phần còn lại: 1.0 - w_rating
    remaining_after_rating = 1.0 - w_rating
        
    # Xử lý lỗi làm tròn số học (floating point error)
    remaining_after_rating = round(remaining_after_rating, 2)
        
    w_revenue = st.slider(
        "💰 2. Doanh thu (Revenue)", 
        min_value=0.0, 
        max_value=remaining_after_rating, 
        value=min(0.2, remaining_after_rating), 
        step=0.05,
        key="slider_revenue"
    )
        
    # --- THANH 3: ĐỘ PHỔ BIẾN ---
    w_vote = 1.0 - w_rating - w_revenue
    w_vote = round(w_vote, 2) # Làm tròn để hiển thị cho đẹp
        
    st.write(f"🔥 **3. Độ phổ biến: {w_vote}**")
        
    # Progress bar nhận giá trị từ 0.0 đến 1.0 nên truyền thẳng w_vote vào
    st.progress(w_vote)

with col_display:
    st.subheader("2. Tỷ lệ phân bổ")
    df_weights = pd.DataFrame({
        'Yếu tố': ['Rating', 'Revenue', 'Popularity'],
        'Trọng số': [w_rating, w_revenue, w_vote]
    })
    fig_pie = px.pie(
        df_weights, values='Trọng số', names='Yếu tố', hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Format hiển thị trên biểu đồ tròn
    fig_pie.update_traces(textinfo='value', texttemplate='%{value:.1f}')
    fig_pie.update_layout(showlegend=False, height=250, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
    
if not filtered_df.empty:
    # 1. Chuẩn hóa dữ liệu đầu vào (Min-Max Scaling)
    df_score = filtered_df.copy()
    df_score['$Worldwide'] = df_score['$Worldwide'].fillna(0)
    df_score['Vote_Count'] = df_score['Vote_Count'].fillna(0)

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if (series.max() - series.min()) > 0 else 0

    df_score['norm_rating'] = normalize(df_score['Rating'])
    df_score['norm_revenue'] = normalize(df_score['$Worldwide'])
    df_score['norm_vote'] = normalize(df_score['Vote_Count'])

    # Tính Final Score (Thang 0-1)
    df_score['Final_Score'] = (
        (df_score['norm_rating'] * w_rating) + 
        (df_score['norm_revenue'] * w_revenue) + 
        (df_score['norm_vote'] * w_vote)
    )

    #  Sắp xếp
    df_ranked = df_score.sort_values(by='Final_Score', ascending=False).head(20)

    # Biểu đồ đóng góp
    st.subheader(f"🥇 Top 20 Phim (Thang 0 - 1)")
        
    df_viz = df_ranked[['Title', 'norm_rating', 'norm_revenue', 'norm_vote', 'Final_Score']].copy()
        
    # Nhân trực tiếp
    df_viz['Rating'] = df_viz['norm_rating'] * w_rating
    df_viz['Revenue'] = df_viz['norm_revenue'] * w_revenue
    df_viz['Popularity'] = df_viz['norm_vote'] * w_vote
        
    fig_rank = px.bar(
        df_viz, 
        x=['Rating', 'Revenue', 'Popularity'], 
        y='Title', 
        orientation='h',
        labels={'value': 'Điểm số (0-1)', 'variable': 'Yếu tố'},
        height=600
    )
    fig_rank.update_layout(
        yaxis={'categoryorder':'total ascending'}, 
        xaxis=dict(range=[0, 1]), # Cố định trục X max là 1.0
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Bảng chi tiết
    st.dataframe(
        df_ranked[['Title', 'Year', 'Rating', '$Worldwide', 'Vote_Count', 'Final_Score']],
        column_config={
            "Final_Score": st.column_config.ProgressColumn(
                "Điểm tổng hợp", 
                format="%.2f",    
                min_value=0, 
                max_value=1       # Max là 1.0
            ),
            "$Worldwide": st.column_config.NumberColumn("Doanh thu", format="$%.2f"),
            "Rating": st.column_config.NumberColumn("Rating gốc", format="%.1f"),
        },
        use_container_width=True
    )
else:
    st.warning("Không có dữ liệu phim.")
    
st.markdown("---")

st.title("Dashboard Phân tích Phim 🎬")    
genres = st.multiselect("🎭 Thể loại (Lọc chung)", options=unique_genres, default=unique_genres[:3])
countries = st.multiselect("🌐 Quốc gia", options=unique_countries, default=[])
