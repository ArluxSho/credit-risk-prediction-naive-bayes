from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# KONFIGURASI 

NAMA = "Aisyah Nurul Sholikhah"
NIM = "L0124085"
MATA_KULIAH = "Data Mining"
PROGRAM_STUDI = "Informatika / UNS"

DATA_FILE = "credit_risk_dataset.csv"
GUNAKAN_ATURAN_KONSISTENSI = True   # sama dengan DROP_RULES di notebook; bisa diubah juga lewat sidebar
RANDOM_STATE = 42


# KONSTANTA

TEXT_COLS = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
LOG_COLS = ["person_income", "loan_amnt"]
IMP_COLS = ["person_emp_length", "loan_int_rate"]
DTI_BINS = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
HOME_OPTIONS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT_OPTIONS = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
REQUIRED_COLS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent",
    "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
]
FACTOR_LABEL = {
    "loan_percent_income": "Rasio pinjaman terhadap pendapatan",
    "person_home_ownership": "Status kepemilikan rumah",
    "loan_grade": "Grade pinjaman",
    "person_income": "Pendapatan tahunan",
    "loan_intent": "Tujuan pinjaman",
    "loan_int_rate": "Suku bunga",
    "loan_amnt": "Jumlah pinjaman",
    "person_emp_length": "Lama bekerja",
    "cb_person_default_on_file": "Riwayat gagal bayar",
    "person_age": "Usia",
    "cb_person_cred_hist_length": "Panjang riwayat kredit",
}
ORDINAL = ["pertama", "kedua", "ketiga", "keempat", "kelima", "keenam", "ketujuh",
           "kedelapan", "kesembilan", "kesepuluh", "kesebelas"]
KONSISTENSI = "Rasio pinjaman tidak sesuai loan_amnt / person_income"

PRESETS = {
    "tinggi": dict(in_age=25, in_income=9600, in_emp=1.0, in_home="RENT", in_intent="MEDICAL",
                   in_grade="C", in_amnt=5500, in_rate=12.87, in_hist="N", in_cred=3),
    "rendah": dict(in_age=35, in_income=85000, in_emp=8.0, in_home="MORTGAGE", in_intent="VENTURE",
                   in_grade="A", in_amnt=8000, in_rate=7.50, in_hist="N", in_cred=10),
}

PRIMARY, ACCENT, MUTED = "#2F5D8A", "#C0504D", "#AEB8C4"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 110,
})

st.set_page_config(page_title="Prediksi Risiko Gagal Bayar", layout="wide")



# FUNGSI BANTU

def num(x, d=1):
    """Angka gaya Indonesia: 1.234,5"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def ribu(n):
    return f"{int(n):,}".replace(",", ".")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def audit_rules(d):
    return {
        "Usia > 80 tahun": d["person_age"] > 80,
        "Masa kerja > 60 tahun": d["person_emp_length"] > 60,
        "Usia - masa kerja < 14 (mulai kerja terlalu dini)": (d["person_age"] - d["person_emp_length"]) < 14,
        "Riwayat kredit dimulai sebelum usia 16": (d["person_age"] - d["cb_person_cred_hist_length"]) < 16,
        KONSISTENSI: (d["loan_amnt"] / d["person_income"] - d["loan_percent_income"]).abs() > 0.01,
        "Pendapatan atau jumlah pinjaman <= 0": (d["person_income"] <= 0) | (d["loan_amnt"] <= 0),
    }


def make_pipeline():
    imputer = ColumnTransformer(
        [("imp", SimpleImputer(strategy="median"), IMP_COLS)],
        remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    logger = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), LOG_COLS)],
        remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("imputer", imputer), ("log", logger), ("gnb", GaussianNB())])


def in_bin(series, lo, hi):
    left = series >= lo if lo == DTI_BINS[0] else series > lo
    return left & (series <= hi)


def dti_table(df):
    rows = []
    for lo, hi in zip(DTI_BINS[:-1], DTI_BINS[1:]):
        m = in_bin(df["loan_percent_income"], lo, hi)
        if m.sum() > 0:
            rows.append({"Rentang": f"{lo:.2f} - {hi:.2f}", "lo": lo, "hi": hi, "Jumlah": int(m.sum()),
                         "Gagal_bayar_pct": df.loc[m, "loan_status"].mean() * 100})
    return pd.DataFrame(rows).set_index("Rentang")


def dti_cut(df):
    """Batas rasio tempat gagal bayar melonjak paling tajam."""
    t = dti_table(df)
    i = int(np.argmax(np.diff(t["Gagal_bayar_pct"].values)))
    return float(t["hi"].iloc[i])


def rate_by(df, col):
    return (df.groupby(col)["loan_status"].mean() * 100).sort_values(ascending=False)



# DATA

@st.cache_data(show_spinner=False)
def read_csv_path(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_csv_bytes(content):
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner="Membersihkan data...")
def clean_data(df_raw, use_consistency_rule):
    d = df_raw.copy()
    for c in TEXT_COLS:
        d[c] = d[c].astype(str).str.strip().str.upper()
    log = [("Data awal", 0, len(d))]
    n = len(d)
    d = d.drop_duplicates()
    log.append(("Duplikat persis", n - len(d), len(d)))
    for name in list(audit_rules(d)):
        if name == KONSISTENSI and not use_consistency_rule:
            continue
        mask = audit_rules(d)[name]
        log.append((name, int(mask.sum()), len(d) - int(mask.sum())))
        d = d[~mask]
    return d.reset_index(drop=True), pd.DataFrame(log, columns=["Tahap", "Baris dibuang", "Sisa baris"])


def encode(df_clean):
    d = df_clean.copy()
    d["loan_grade"] = d["loan_grade"].map(GRADE_MAP)
    d["cb_person_default_on_file"] = d["cb_person_default_on_file"].map({"N": 0, "Y": 1})
    d = pd.get_dummies(d, columns=["person_home_ownership", "loan_intent"], drop_first=True, dtype=int)
    return d.drop(columns="loan_status"), d["loan_status"]



# PELATIHAN DAN EVALUASI (di-cache)

@st.cache_resource(show_spinner="Melatih dan mengevaluasi model (10-fold). Proses pertama sekitar 10-30 detik...")
def train_and_evaluate(df_clean):
    X, y = encode(df_clean)
    pipe = make_pipeline()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
    fold_df = pd.DataFrame({m: cv["test_" + m] for m in scoring}, index=[f"Fold-{i}" for i in range(1, 11)])

    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    report = pd.DataFrame(classification_report(
        y, pred, target_names=["Lancar (0)", "Gagal bayar (1)"], output_dict=True)).T
    fpr, tpr, _ = roc_curve(y, proba)
    thr_rows = []
    for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]:
        p, r, f, _ = precision_recall_fscore_support(y, (proba >= t).astype(int), average="binary", zero_division=0)
        thr_rows.append({"Ambang": t, "Precision": p, "Recall": r, "F1": f})

    # grouped permutation importance pada data uji tiap fold
    groups = {
        "person_home_ownership": [c for c in X.columns if c.startswith("person_home_ownership_")],
        "loan_intent": [c for c in X.columns if c.startswith("loan_intent_")],
    }
    grouped = {c for v in groups.values() for c in v}
    groups.update({c: [c] for c in X.columns if c not in grouped})
    rng = np.random.default_rng(RANDOM_STATE)
    imp = {g: [] for g in groups}
    for tr, te in skf.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        Xte, yte = X.iloc[te], y.iloc[te]
        base = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        for g, cols in groups.items():
            drops = []
            for _ in range(5):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].values[rng.permutation(len(Xte))]
                drops.append(base - roc_auc_score(yte, m.predict_proba(Xp)[:, 1]))
            imp[g].append(np.mean(drops))
    imp_df = (pd.DataFrame({"Faktor": list(imp), "Penurunan_AUC": [np.mean(v) for v in imp.values()],
                            "Std": [np.std(v) for v in imp.values()]})
              .sort_values("Penurunan_AUC", ascending=False).reset_index(drop=True))

    smd = ((X[y == 1].mean() - X[y == 0].mean()) / X.std()).rename("Selisih (satuan std)")
    smd = smd.reindex(smd.abs().sort_values(ascending=False).index)

    return {
        "model": make_pipeline().fit(X, y), "columns": list(X.columns), "fold_df": fold_df,
        "summary": pd.DataFrame({"Rata-rata": fold_df.mean(), "Std": fold_df.std()}),
        "cm": cm, "report": report, "fpr": fpr, "tpr": tpr, "auc_oof": roc_auc_score(y, proba),
        "thresholds": pd.DataFrame(thr_rows), "importance": imp_df, "smd": smd,
        "baseline": max(y.mean(), 1 - y.mean()), "pos_rate": y.mean(), "n": len(y),
    }



# PENJELASAN FAKTOR (dihitung dari data, bukan diketik manual)

def factor_context(df):
    g = df["loan_grade"].map(GRADE_MAP)
    h = df["cb_person_default_on_file"].map({"N": 0, "Y": 1})
    return {
        "cut": dti_cut(df),
        "r_grade_rate": g.corr(df["loan_int_rate"]),
        "r_amnt_lpi": df["loan_amnt"].corr(df["loan_percent_income"]),
        "r_age_cred": df["person_age"].corr(df["cb_person_cred_hist_length"]),
        "r_hist_grade": h.corr(g),
    }


def importance_tail(rank, val):
    """Kalimat penutup yang mengikuti besar penurunan AUC sebenarnya."""
    if rank == 0:
        return " Saat faktor ini diacak, AUC turun paling besar, sehingga model paling bergantung padanya."
    if val < 0.002:
        return " Kontribusi tambahannya nyaris nol, sehingga model hampir tidak memakainya."
    if val < 0.015:
        return " Kontribusi tambahannya relatif kecil."
    return " Kontribusi tambahannya cukup besar."


def explain_factor(f, df, ctx, rank, val):
    """Alasan (diawali huruf kecil) mengapa faktor berada di peringkatnya; angka dihitung dari data."""
    bad = df["loan_status"] == 1
    mean = lambda c: (df.loc[bad, c].mean(), df.loc[~bad, c].mean())
    tail = importance_tail(rank, val)
    if f == "loan_percent_income":
        a, b = mean(f)
        cut = ctx["cut"]
        hi = df.loc[df[f] > cut, "loan_status"].mean() * 100
        lo = df.loc[df[f] <= cut, "loan_status"].mean() * 100
        txt = (f"peminjam gagal bayar rata-rata meminjam {num(a * 100, 0)}% dari pendapatan tahunannya, sedangkan "
               f"peminjam lancar {num(b * 100, 0)}%. Gagal bayar mencapai {num(hi)}% pada rasio di atas "
               f"{num(cut * 100, 0)}% dibanding {num(lo)}% pada rasio sampai {num(cut * 100, 0)}%. Rasio ini langsung "
               f"menggambarkan beban pinjaman terhadap kemampuan bayar.")
    elif f == "person_home_ownership":
        r = rate_by(df, f)
        txt = (f"penyewa (RENT) gagal bayar {num(r.get('RENT', np.nan))}%, dibanding pemegang KPR "
               f"({num(r.get('MORTGAGE', np.nan))}%) dan pemilik rumah ({num(r.get('OWN', np.nan))}%). Seluruh "
               f"kategori status rumah diacak bersama sebagai satu faktor, sehingga pengaruhnya terukur penuh.")
    elif f == "loan_grade":
        g = df["loan_grade"].map(GRADE_MAP)
        low = df.loc[g <= 2, "loan_status"].mean() * 100
        high = df.loc[g >= 3, "loan_status"].mean() * 100
        txt = (f"gagal bayar rata-rata {num(low)}% pada Grade A sampai C dan {num(high)}% pada Grade D sampai G. "
               f"Grade adalah penilaian risiko awal dari pemberi pinjaman sehingga sudah merangkum banyak informasi, "
               f"dan sebagian informasinya sama dengan suku bunga (korelasi r = {num(ctx['r_grade_rate'], 2)}).")
    elif f == "person_income":
        a, b = df.loc[bad, f].median(), df.loc[~bad, f].median()
        txt = (f"median pendapatan peminjam gagal bayar {num(a, 0)} berbanding {num(b, 0)} pada peminjam lancar. "
               f"Tingkat pendapatan menentukan kemampuan menyerap cicilan, di luar informasi yang sudah dimuat rasio.")
    elif f == "loan_intent":
        r = rate_by(df, f)
        txt = (f"gagal bayar tertinggi pada {r.index[0]} ({num(r.iloc[0])}%) dan terendah pada {r.index[-1]} "
               f"({num(r.iloc[-1])}%).")
    elif f == "loan_int_rate":
        a, b = mean(f)
        txt = (f"suku bunga peminjam gagal bayar rata-rata {num(a, 2)}% berbanding {num(b, 2)}%, tetapi hampir "
               f"identik dengan grade (r = {num(ctx['r_grade_rate'], 2)}). Saat suku bunga diacak, grade masih membawa "
               f"informasi yang sama, sehingga penurunan AUC tidak mencerminkan kekuatan hubungan tunggalnya.")
    elif f == "loan_amnt":
        a, b = mean(f)
        txt = (f"jumlah pinjaman rata-rata {num(a, 0)} (gagal bayar) berbanding {num(b, 0)} (lancar). Informasinya "
               f"sebagian sudah terwakili oleh rasio pinjaman terhadap pendapatan (r = {num(ctx['r_amnt_lpi'], 2)}).")
    elif f == "person_emp_length":
        a, b = mean(f)
        txt = f"lama bekerja rata-rata {num(a, 1)} tahun (gagal bayar) berbanding {num(b, 1)} tahun (lancar)."
    elif f == "cb_person_default_on_file":
        r = rate_by(df, f)
        txt = (f"peminjam bercatatan gagal bayar (Y) gagal bayar {num(r.get('Y', np.nan))}% berbanding "
               f"{num(r.get('N', np.nan))}% pada yang tidak, dan faktor ini berkorelasi dengan grade "
               f"(r = {num(ctx['r_hist_grade'], 2)}).")
    elif f == "person_age":
        a, b = mean(f)
        txt = f"usia rata-rata {num(a, 1)} tahun (gagal bayar) berbanding {num(b, 1)} tahun (lancar)."
    elif f == "cb_person_cred_hist_length":
        a, b = mean(f)
        txt = (f"panjang riwayat kredit rata-rata {num(a, 1)} tahun (gagal bayar) berbanding {num(b, 1)} tahun (lancar), "
               f"dan berkorelasi kuat dengan usia (r = {num(ctx['r_age_cred'], 2)}).")
    else:
        txt = "faktor ini dihitung dari data."
    return txt + tail



# GRAFIK

def bar_chart(series, title, ylabel, horizontal=False, highlight=None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [ACCENT if (highlight and i in highlight) else PRIMARY for i in series.index]
    if horizontal:
        bars = ax.barh(list(series.index[::-1]), series.values[::-1], color=colors[::-1])
        ax.set_xlabel(ylabel)
        ax.grid(axis="y", visible=False)
        for b in bars:
            ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, max(series.values) * 1.18)
    else:
        bars = ax.bar([str(i) for i in series.index], series.values, color=colors)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", fontsize=8)
        ax.set_ylim(0, max(series.values) * 1.18)
        if len(series) > 5:
            ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def cm_chart(cm):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Lancar", "Gagal bayar"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Prediksi {l}" for l in labels])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Aktual {l}" for l in labels])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, ribu(cm[i, j]), ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix (10-fold)")
    fig.tight_layout()
    return fig


def roc_chart(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, color=PRIMARY, lw=2, label=f"Gaussian NB (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Acak")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_title("Kurva ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def importance_chart(imp_df):
    d = imp_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh([FACTOR_LABEL.get(f, f) for f in d["Faktor"]], d["Penurunan_AUC"], xerr=d["Std"],
            color=PRIMARY, error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xlabel("Penurunan ROC-AUC saat faktor diacak")
    ax.set_title("Peringkat faktor (grouped permutation importance)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def prob_chart(p):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    bars = ax.barh(["Gagal bayar", "Lancar"], [p[1] * 100, p[0] * 100], color=[ACCENT, PRIMARY])
    ax.set_xlim(0, 115)
    ax.set_xlabel("Skor model (%)")
    ax.grid(axis="y", visible=False)
    for b in bars:
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig



# SIDEBAR DAN PEMUATAN DATA

def sidebar():
    with st.sidebar:
        st.markdown("### Identitas")
        st.markdown(f"**{NAMA}**  \n{NIM}  \n{MATA_KULIAH}  \n{PROGRAM_STUDI}")
        st.divider()
        st.markdown("### Pengaturan data")
        use_rule = st.checkbox(
            "Buang baris dengan rasio pinjaman tidak konsisten", value=GUNAKAN_ATURAN_KONSISTENSI,
            help="Membuang baris jika loan_percent_income tidak sesuai loan_amnt / person_income. "
                 "Ubah untuk melihat sensitivitas hasil. Model akan dilatih ulang.")
        st.divider()
        st.caption("Gaussian Naive Bayes, 10-Fold Stratified Cross Validation, random_state = 42.")
    return use_rule


def load_raw():
    path = Path(__file__).parent / DATA_FILE
    if path.exists():
        df = read_csv_path(str(path))
    else:
        st.warning(f"Berkas {DATA_FILE} tidak ditemukan di folder aplikasi. Unggah berkas CSV dataset di bawah ini.")
        up = st.file_uploader("Credit Risk Dataset (CSV)", type="csv")
        if up is None:
            st.stop()
        df = read_csv_bytes(up.getvalue())
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("Kolom berikut tidak ada pada dataset: " + ", ".join(missing))
        st.stop()
    return df



# TAB

def tab_ringkasan(df_raw, df_clean, log_df, res):
    st.subheader("Ringkasan")
    st.write("Aplikasi ini memprediksi apakah pengajuan pinjaman berpotensi lancar (kelas 0) atau gagal bayar (kelas 1) "
             "dengan Gaussian Naive Bayes. Model dipakai sebagai alat pemilah awal (early screening), "
             "bukan pengambil keputusan tunggal.")

    s = res["summary"]["Rata-rata"]
    c = st.columns(5)
    c[0].metric("Accuracy", f"{num(s['accuracy'] * 100, 2)}%", f"baseline {num(res['baseline'] * 100, 2)}%", delta_color="off")
    c[1].metric("ROC-AUC", f"{s['roc_auc']:.4f}")
    c[2].metric("Recall gagal bayar", f"{num(s['recall'] * 100, 2)}%")
    c[3].metric("Precision", f"{num(s['precision'] * 100, 2)}%")
    c[4].metric("F1-Score", f"{num(s['f1'] * 100, 2)}%")

    left, right = st.columns([1, 1.2])
    with left:
        with st.container(border=True):
            st.markdown("**Dataset**")
            st.markdown(f"- Sumber: Credit Risk Dataset (Kaggle)\n"
                        f"- Data awal: {ribu(len(df_raw))} baris dan {df_raw.shape[1]} kolom\n"
                        f"- Setelah cleaning: {ribu(len(df_clean))} baris\n"
                        f"- Proporsi gagal bayar: {num(res['pos_rate'] * 100, 2)}%")
        with st.container(border=True):
            st.markdown("**Metodologi**")
            st.markdown("1. Audit dan pembersihan data (duplikat dan nilai tidak masuk akal)\n"
                        "2. Encoding kategori, lalu imputasi median dan transformasi log di dalam pipeline\n"
                        "3. Gaussian Naive Bayes dengan 10-Fold Stratified Cross Validation\n"
                        "4. Evaluasi dan grouped permutation importance")
    with right:
        st.markdown("**Log pembersihan data**")
        st.dataframe(log_df, hide_index=True)


def tab_eda(df_clean):
    st.subheader("Eksplorasi Data")
    st.caption(f"Dihitung dari {ribu(len(df_clean))} baris data bersih.")

    grade = (df_clean.groupby("loan_grade")["loan_status"].mean() * 100).sort_index()
    intent = rate_by(df_clean, "loan_intent")
    home = rate_by(df_clean, "person_home_ownership")
    dti = dti_table(df_clean)
    cut = dti_cut(df_clean)

    r1 = st.columns(2)
    with r1[0]:
        show(bar_chart(grade, "Gagal bayar menurut grade", "Gagal bayar (%)", highlight={"D", "E", "F", "G"}))
    with r1[1]:
        show(bar_chart(intent, "Gagal bayar menurut tujuan pinjaman", "Gagal bayar (%)", horizontal=True))
    r2 = st.columns(2)
    with r2[0]:
        show(bar_chart(home, "Gagal bayar menurut status rumah", "Gagal bayar (%)"))
    with r2[1]:
        show(bar_chart(dti["Gagal_bayar_pct"], "Gagal bayar menurut rasio pinjaman/pendapatan", "Gagal bayar (%)",
                       highlight={i for i, r in dti.iterrows() if r["lo"] >= cut - 1e-9}))

    st.markdown("**Rata-rata fitur menurut status pinjaman**")
    cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
    mean_df = df_clean.groupby("loan_status")[cols].mean().round(2).T
    mean_df.columns = ["Lancar (0)", "Gagal bayar (1)"]
    st.dataframe(mean_df)


def tab_evaluasi(res):
    st.subheader("Evaluasi Model")
    s = res["summary"]
    acc = s.loc["accuracy", "Rata-rata"]
    st.write(f"Hasil 10-Fold Stratified Cross Validation pada {ribu(res['n'])} data. Accuracy {num(acc * 100, 2)}% "
             f"berada {num((acc - res['baseline']) * 100, 1)} poin di atas baseline {num(res['baseline'] * 100, 2)}% "
             f"(menebak kelas mayoritas), sehingga penilaian bertumpu pada recall, F1, dan ROC-AUC.")

    summ = s.copy()
    summ.index = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(summ.round(4))

    c = st.columns(2)
    with c[0]:
        show(cm_chart(res["cm"]))
    with c[1]:
        show(roc_chart(res["fpr"], res["tpr"], res["auc_oof"]))

    cm = res["cm"]
    prec = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    rec = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    st.info(f"Dari {ribu(cm[1].sum())} kasus gagal bayar, {ribu(cm[1, 1])} terdeteksi (recall {num(rec * 100)}%) dan "
            f"{ribu(cm[1, 0])} lolos. Dari {ribu(cm[:, 1].sum())} pemohon yang ditandai berisiko, {ribu(cm[0, 1])} "
            f"sebenarnya lancar (sekitar {num((1 - prec) * 100, 0)}% false alarm).")

    with st.expander("Hasil per fold"):
        f = (res["fold_df"] * 100).round(2)
        f["roc_auc"] = res["fold_df"]["roc_auc"].round(4)
        st.dataframe(f)
    with st.expander("Classification report"):
        st.dataframe(res["report"].round(4))
    with st.expander("Pengaruh ambang keputusan"):
        st.caption("Ambang rendah menangkap lebih banyak gagal bayar tetapi menambah false alarm.")
        st.dataframe(res["thresholds"].round(4), hide_index=True)


def tab_faktor(df_clean, res):
    st.subheader("Faktor yang Paling Berpengaruh")
    st.write("Peringkat dihitung dengan grouped permutation importance pada data uji tiap fold: nilai satu faktor diacak, "
             "lalu diukur seberapa besar ROC-AUC turun. Kolom dummy dari variabel yang sama (status rumah, tujuan "
             "pinjaman) diacak bersama sebagai satu faktor.")
    imp = res["importance"]
    ctx = factor_context(df_clean)

    left, right = st.columns([1.1, 1])
    with left:
        show(importance_chart(imp))
    with right:
        t = imp.copy()
        t.insert(0, "Peringkat", range(1, len(t) + 1))
        t["Faktor"] = t["Faktor"].map(lambda f: FACTOR_LABEL.get(f, f))
        t["Penurunan_AUC"] = t["Penurunan_AUC"].round(4)
        t["Std"] = t["Std"].round(4)
        st.dataframe(t, hide_index=True)

    st.markdown("**Mengapa urutannya demikian**")
    for i, row in imp.iterrows():
        f = row["Faktor"]
        urut = ORDINAL[i] if i < len(ORDINAL) else f"ke-{i + 1}"
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {FACTOR_LABEL.get(f, f)}** (penurunan AUC {row['Penurunan_AUC']:.4f})  \n"
                        f"Berada di urutan {urut} karena {explain_factor(f, df_clean, ctx, i, row['Penurunan_AUC'])}")

    with st.expander("Pembanding: selisih rata-rata fitur antar kelas"):
        st.caption("Dalam satuan simpangan baku. Positif berarti fitur lebih tinggi pada kelas gagal bayar. Ukuran ini "
                   "menunjukkan kekuatan hubungan tunggal, berbeda dengan importance yang mengukur kontribusi tambahan.")
        st.dataframe(res["smd"].round(3).to_frame())
    st.caption("Importance menunjukkan kontribusi prediktif bagi model, bukan hubungan sebab-akibat. Grade dan suku bunga "
               "ditetapkan oleh pemberi pinjaman sehingga sebagian sudah mencerminkan penilaian risiko.")


def tab_rekomendasi(df_clean, log_df, res):
    st.subheader("Rekomendasi untuk Manajemen")
    g = df_clean.groupby("loan_grade")["loan_status"].mean() * 100
    h = rate_by(df_clean, "person_home_ownership")
    it = rate_by(df_clean, "loan_intent")
    dti = dti_table(df_clean)
    cut = dti_cut(df_clean)
    top = [FACTOR_LABEL.get(f, f) for f in res["importance"]["Faktor"].head(3)]
    s = res["summary"]["Rata-rata"]
    below = dti.loc[np.isclose(dti["hi"], cut), "Gagal_bayar_pct"].iloc[0]
    above = dti.loc[np.isclose(dti["lo"], cut), "Gagal_bayar_pct"].iloc[0]
    removed = int(log_df["Baris dibuang"].sum())

    rows = [
        ("Tiga faktor teratas model", ", ".join(top),
         "Jadikan ketiganya kriteria utama penilaian dan pantau secara berkala."),
        ("Batas rasio pinjaman terhadap pendapatan",
         f"Gagal bayar {num(below)}% pada rasio sampai {num(cut, 2)} dan melonjak ke {num(above)}% sesudahnya.",
         f"Tetapkan batas sekitar {num(cut * 100, 0)}% dan wajibkan agunan atau persetujuan berjenjang di atasnya."),
        ("Pengetatan pada grade berisiko",
         f"Gagal bayar maksimal {num(g.loc[['A', 'B', 'C']].max())}% pada Grade A sampai C, "
         f"{num(g.get('D', np.nan))}% pada Grade D, {num(g.get('G', np.nan))}% pada Grade G.",
         "Perketat persetujuan mulai Grade D: agunan tambahan, limit lebih rendah, atau persetujuan berjenjang."),
        ("Status tempat tinggal",
         f"Penyewa {num(h.get('RENT', np.nan))}% berbanding KPR {num(h.get('MORTGAGE', np.nan))}% "
         f"dan pemilik {num(h.get('OWN', np.nan))}%.",
         "Perlakukan status sewa sebagai sinyal tambahan bersama rasio dan grade, bukan alasan tolak otomatis."),
        ("Tujuan pinjaman",
         f"Tertinggi {it.index[0]} ({num(it.iloc[0])}%), terendah {it.index[-1]} ({num(it.iloc[-1])}%).",
         "Terapkan pemeriksaan tambahan pada tujuan berisiko dan tinjau ulang kebijakan produknya."),
        ("Pemakaian model",
         f"ROC-AUC {num(s['roc_auc'], 3)}, recall {num(s['recall'] * 100)}%, precision {num(s['precision'] * 100)}% "
         f"(sekitar {num(100 - s['precision'] * 100, 0)}% false alarm).",
         "Gunakan sebagai early screening, keputusan akhir tetap oleh analis, dan atur ambang sesuai toleransi risiko."),
        ("Kualitas data",
         f"{ribu(removed)} baris duplikat atau tidak masuk akal ditemukan pada data awal.",
         "Tambahkan validasi saat input (rentang usia, masa kerja, kesesuaian rasio) dan pencegahan duplikat."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Fokus", "Temuan data", "Tindakan yang disarankan"]), hide_index=True)


def set_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def tab_simulasi(df_clean, res):
    st.subheader("Simulasi Prediksi")
    st.write("Isi profil pemohon, atau mulai dari salah satu contoh, lalu jalankan prediksi.")

    if "in_age" not in st.session_state:
        set_preset("tinggi")
    b = st.columns([1, 1, 3])
    b[0].button("Contoh risiko tinggi", on_click=set_preset, args=("tinggi",))
    b[1].button("Contoh risiko rendah", on_click=set_preset, args=("rendah",))

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profil dan pekerjaan**")
            age = st.number_input("Usia (tahun)", min_value=18, max_value=80, step=1, key="in_age")
            income = st.number_input("Pendapatan tahunan", min_value=1000, max_value=2000000, step=500, key="in_income")
            emp = st.number_input("Lama bekerja (tahun)", min_value=0.0, max_value=60.0, step=0.5, key="in_emp")
            home = st.selectbox("Status rumah", HOME_OPTIONS, key="in_home")
        with c2:
            st.markdown("**Detail pinjaman**")
            intent = st.selectbox("Tujuan pinjaman", INTENT_OPTIONS, key="in_intent")
            grade = st.selectbox("Grade pinjaman", list(GRADE_MAP), key="in_grade")
            amnt = st.number_input("Jumlah pinjaman", min_value=500, max_value=50000, step=500, key="in_amnt")
            rate = st.number_input("Suku bunga (%)", min_value=5.0, max_value=25.0, step=0.1, format="%.2f", key="in_rate")
        with c3:
            st.markdown("**Riwayat kredit**")
            hist = st.selectbox("Pernah gagal bayar?", ["N", "Y"], key="in_hist",
                                format_func=lambda x: "Tidak" if x == "N" else "Ya")
            cred = st.number_input("Panjang riwayat kredit (tahun)", min_value=1, max_value=35, step=1, key="in_cred")
            st.caption("Rasio pinjaman terhadap pendapatan dihitung otomatis dari jumlah pinjaman dan pendapatan.")
        submit = st.form_submit_button("Jalankan prediksi", type="primary")

    if not submit:
        return

    lpi = round(amnt / income, 2)
    row = dict.fromkeys(res["columns"], 0)
    row.update({
        "person_age": age, "person_income": income, "person_emp_length": emp, "loan_grade": GRADE_MAP[grade],
        "loan_amnt": amnt, "loan_int_rate": rate, "loan_percent_income": lpi,
        "cb_person_default_on_file": 1 if hist == "Y" else 0, "cb_person_cred_hist_length": cred,
    })
    for prefix, val in [("person_home_ownership", home), ("loan_intent", intent)]:
        if f"{prefix}_{val}" in row:      # kategori acuan (MORTGAGE, DEBTCONSOLIDATION) tidak punya kolom
            row[f"{prefix}_{val}"] = 1
    p = res["model"].predict_proba(pd.DataFrame([row])[res["columns"]])[0]
    kelas = int(p[1] >= 0.5)

    st.divider()
    st.markdown("#### Hasil klasifikasi")
    left, right = st.columns([1.2, 1])
    with left:
        if kelas == 1:
            st.error("**Kelas 1: gagal bayar (risiko tinggi)**")
        else:
            st.success("**Kelas 0: lancar (risiko rendah)**")
        st.write(f"Rasio pinjaman terhadap pendapatan: **{num(lpi * 100, 0)}%**. "
                 f"Skor risiko gagal bayar: **{num(p[1] * 100, 2)}%**.")
        st.caption("Skor Naive Bayes cenderung ekstrem karena fitur saling berkorelasi, sehingga dibaca sebagai "
                   "peringkat risiko, bukan peluang literal.")

        lo_hi = next(((lo, hi) for lo, hi in zip(DTI_BINS[:-1], DTI_BINS[1:]) if lpi <= hi), None)
        dti_rate = (df_clean.loc[in_bin(df_clean["loan_percent_income"], *lo_hi), "loan_status"].mean() * 100
                    if lo_hi else np.nan)
        base = df_clean["loan_status"].mean() * 100
        rows = [
            (f"Grade {grade}", df_clean.loc[df_clean["loan_grade"] == grade, "loan_status"].mean() * 100),
            (f"Status rumah {home}", df_clean.loc[df_clean["person_home_ownership"] == home, "loan_status"].mean() * 100),
            (f"Tujuan {intent}", df_clean.loc[df_clean["loan_intent"] == intent, "loan_status"].mean() * 100),
            ("Rasio " + (f"{lo_hi[0]:.2f} - {lo_hi[1]:.2f}" if lo_hi else "di luar rentang"), dti_rate),
            ("Riwayat gagal bayar " + ("Ya" if hist == "Y" else "Tidak"),
             df_clean.loc[df_clean["cb_person_default_on_file"] == hist, "loan_status"].mean() * 100),
        ]
        ctx_df = pd.DataFrame(rows, columns=["Kategori pemohon", "Gagal bayar historis (%)"]).round(2)
        ctx_df["Dibanding rata-rata"] = ctx_df["Gagal bayar historis (%)"].map(
            lambda v: "-" if pd.isna(v) else ("lebih tinggi" if v > base else "lebih rendah"))
        st.markdown(f"**Gagal bayar historis pada kategori pemohon** (rata-rata seluruh data {num(base, 2)}%)")
        st.dataframe(ctx_df, hide_index=True)
    with right:
        show(prob_chart(p))


# TAB 6: GITHUB


# GITHUB DATA
GITHUB_URL = "https://github.com/ArluxSho/credit-risk-prediction-naive-bayes"
GITHUB_USERNAME = "ArluxSho"
GITHUB_REPOSITORY = "credit-risk-prediction-naive-bayes"


def tab_github ():

    st.subheader("Repository GitHub")

    st.write(
        "Repository ini berisi source code dan dokumentasi "
        "implementasi klasifikasi risiko gagal bayar pinjaman "
        "menggunakan algoritma Gaussian Naive Bayes."
    )

    st.divider()

    # Informasi repository
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Informasi Proyek**")

        st.write("**Repository**")
        st.write(GITHUB_REPOSITORY)

        st.write("**Pemilik**")
        st.write(GITHUB_USERNAME)

        st.write("**Algoritma**")
        st.write("Gaussian Naive Bayes")

        st.write("**Bahasa Pemrograman**")
        st.write("Python")

    with col2:
        st.markdown("**Akses Repository**")

        st.markdown(
            f"""
            <a href="{GITHUB_URL}" target="_blank"
               style="
                   display: inline-block;
                   background-color: #24292f;
                   color: white;
                   padding: 10px 20px;
                   border-radius: 7px;
                   text-decoration: none;
                   font-weight: 600;
                   font-size: 14px;
               ">
                Open GitHub Repository
            </a>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Klik tombol untuk melihat source code, "
            "dokumentasi, dan file proyek."
        )




# MAIN

def main():
    st.title("Klasifikasi Risiko Gagal Bayar Pinjaman")
    st.caption("Gaussian Naive Bayes dengan validasi 10-Fold Stratified Cross Validation")

    use_rule = sidebar()
    df_raw = load_raw()
    df_clean, log_df = clean_data(df_raw, use_rule)
    res = train_and_evaluate(df_clean)

    tabs = st.tabs(["Ringkasan", "Eksplorasi Data", "Evaluasi Model", "Faktor Berpengaruh",
                    "Rekomendasi", "Simulasi Prediksi", "Github Repository"])
    with tabs[0]:
        tab_ringkasan(df_raw, df_clean, log_df, res)
    with tabs[1]:
        tab_eda(df_clean)
    with tabs[2]:
        tab_evaluasi(res)
    with tabs[3]:
        tab_faktor(df_clean, res)
    with tabs[4]:
        tab_rekomendasi(df_clean, log_df, res)
    with tabs[5]:
        tab_simulasi(df_clean, res)
    with tabs[6]:
        tab_github()


main()