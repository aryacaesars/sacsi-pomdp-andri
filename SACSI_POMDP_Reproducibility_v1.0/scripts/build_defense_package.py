"""Build Module 9C reviewer and defense artifacts from frozen 9A/9B evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import load_dashboard_release, sha256_file  # noqa: E402


RESULTS = ROOT / "Results" / "Reviewer_Defense"
DOCS = ROOT / "Docs" / "Reviewer_Defense"
DISsertation_RESULTS = ROOT / "Results" / "Dissertation_Evidence"


REVIEWER_ANSWERS = {
    "RA-01": {
        "short": "Empat rumusan masalah dipetakan satu-ke-satu ke T1–T4. Setiap pasangan berakhir pada evidence dan decision rule yang berbeda, sehingga tujuan, eksperimen, dan kesimpulan dapat ditelusuri.",
        "technical": "RM1/T1 mencakup Virtual Garden dan reward; RM2/T2 fair continuous-control benchmark; RM3/T3 kontribusi forecast-memory; RM4/T4 konfirmasi SACSI multi-seed. Tidak ada tujuan tambahan yang tidak memiliki rumusan masalah.",
        "math": "RM_i ↔ T_i untuk i ∈ {1,2,3,4}.",
        "guard": "Pemetaan metodologis tidak otomatis berarti seluruh hipotesis diterima.",
    },
    "RA-02": {
        "short": "Setiap tujuan memiliki indikator primer, indikator sekunder, protokol, jumlah seed atau gate numerik, dan aturan keputusan. Karena itu keberhasilan tidak ditentukan secara subjektif setelah hasil terlihat.",
        "technical": "Primary endpoint adalah Time in Target untuk perbandingan performa. Gate simulator memakai respons fisik dan residual neraca massa, sedangkan inferensi memakai matched seeds, multiplicity correction, effect size, dan confidence interval.",
        "math": "Decision_j = PASS hanya jika seluruh gate g_j(E) memenuhi ambang τ_j yang dipra-spesifikasikan.",
        "guard": "Outcome sekunder tidak boleh menggantikan primary endpoint setelah benchmark dibuka.",
    },
    "RA-03": {
        "short": "Masalah diperlakukan sebagai POMDP karena controller hanya menerima observation terbatas, bukan seluruh state dan proses laten tanah. History dan forecast dipakai sebagai context untuk memperkaya informasi keputusan.",
        "technical": "State internal Virtual Garden berevolusi dengan forcing dan aksi, tetapi policy hanya melihat observation 8-D. LSTM merangkum history kausal dan forecast proxy memberi informasi h+1; activation, performance benefit, dan superiority diuji terpisah.",
        "math": "x_(t+1)=f(x_t,a_t,w_t), o_t=h(x_t,w_t), z_t=LSTM(o_(t-k:t)), c_t=[o_t,z_t,ŵ_(t+1)].",
        "guard": "Argumentasi POMDP dan branch activation bukan bukti otomatis peningkatan performa.",
    },
    "RA-04": {
        "short": "SAC dipilih sebagai anchor stochastic off-policy untuk aksi kontinu. Twin critics, replay learning, dan entropy regularization memberi pembanding yang relevan sebelum context forecast dan memory ditambahkan.",
        "technical": "Fairness mengunci environment, observation, reward, data split, seed, budget, dan checkpoint rule, tetapi mekanisme intrinsik SAC tetap dipertahankan. SAC Basic juga menjadi anchor warm-start untuk varian context.",
        "math": "J(π)=E[Σ_t γ^t(r_t+αH(π(·|s_t)))], dengan α dituning otomatis.",
        "guard": "Pemilihan SAC secara teoritis tidak membuktikan bahwa SAC selalu terbaik.",
    },
    "RA-05": {
        "short": "DDPG mewakili deterministic actor dengan single critic, sedangkan TD3 menguji twin critics, target smoothing, dan delayed policy update. Keduanya membentuk tangga pembanding sebelum SACSI.",
        "technical": "DDPG dan TD3 memakai observation 8-D, aksi 0–5 mm/jam, reward_v4, split, matched seeds, 6,720 interactions, metric engine, dan validation-only checkpoint selection yang sama dengan SAC pada fair benchmark.",
        "math": "DDPG: ∇J≈E[∇_aQ(s,a)∇_θμ_θ(s)]; TD3 target memakai min(Q′_1,Q′_2) dengan target-action noise dan delayed update.",
        "guard": "Fairness menyamakan protokol, bukan menghapus perbedaan mekanisme algoritma.",
    },
    "RA-06": {
        "short": "Forecast h+1 menguji anticipatory context dan LSTM 24 jam menguji temporal memory. Desain 2×2 memisahkan main effect dan interaction; hasil final hanya mendukung memory effect.",
        "technical": "Empat kondisi adalah F0M0, F1M0, F0M1, dan F1M1 pada matched seeds. Intervensi zero, shuffle, reverse, dan no-context memeriksa aktivitas branch; inferensi final memakai exact sign-flip dengan Holm correction.",
        "math": "E_F=½[(Y_10−Y_00)+(Y_11−Y_01)]; E_M=½[(Y_01−Y_00)+(Y_11−Y_10)]; E_F×M=(Y_11−Y_10)−(Y_01−Y_00).",
        "guard": "Standalone forecast benefit dan forecast-memory synergy tidak didukung.",
    },
    "RA-07": {
        "short": "Virtual Garden adalah simulator neraca air untuk hortikultura generik, bukan digital twin tanaman atau lokasi tertentu. Validasi menunjukkan konsistensi numerik dan respons kontrol yang diharapkan.",
        "technical": "Enam simple cases menguji pengeringan, hujan, irigasi, batas atas, recovery, dan forecast response. Episode DRY/WET/MIXED mempertahankan konfigurasi yang sama serta mengaudit runoff, drainage, dan mass balance.",
        "math": "S_(t+1)=S_t+P_t+I_t−ET_t−R_t−D_t, dengan residual numerik diaudit terhadap gate 10⁻⁸ mm.",
        "guard": "Simulator validation tidak boleh disebut field validation atau crop digital twin validation.",
    },
    "RA-08": {
        "short": "Meteorological forcing adalah data real/raw; soil moisture dan trajectory merupakan keluaran simulasi; forecast final adalah controlled synthetic forecast proxy SF-20. Ketiga kelas evidence selalu dibedakan.",
        "technical": "Weather observations memaksa Virtual Garden. State tanah dihitung dari water balance. Forecast h+1 precipitation, ET0, dan temperature dibuat secara kausal per tahun dengan perturbasi terkontrol, bukan archived as-issued forecast.",
        "math": "E = {M_raw, S_simulated, F_controlled-proxy}; ketiga himpunan tidak dipertukarkan dalam klaim.",
        "guard": "Jangan menyebut seluruh input sebagai field measurements atau operational forecast.",
    },
    "RA-09": {
        "short": "Optimasi adalah proses belajar policy terhadap reward multi-objective, sedangkan efisiensi adalah outcome fisik. Controller tidak disebut efisien hanya karena memakai sedikit air bila gagal mempertahankan target.",
        "technical": "reward_v4 menimbang tracking deficit/surplus, irrigation, action change, dan violation. Pemilihan memakai Pareto Time in Target–water dan stability pada validation 2024; cumulative reward bukan ranking final.",
        "math": "L_t=100(2d_t+s_t)+0.01I_t+0.01|I_t−I_(t−1)|+2V_t; r_t=2−L_t.",
        "guard": "Low irrigation akibat under-irrigation bukan bukti efisiensi.",
    },
    "RA-10": {
        "short": "Fair comparison disusun bertahap: simulator, controller referensi, DDPG–TD3–SAC, SAC-family 2×2, lalu SACSI confirmatory. Setiap tingkat menjawab pertanyaan berbeda.",
        "technical": "Main comparators memakai total 6,720 interactions. Context variants memakai SAC anchor 6,720 ditambah adaptation 6,720; karena itu hasil final membandingkan locked pipelines, bukan arsitektur dengan total budget identik.",
        "math": "Untuk DDPG/TD3/SAC: E,R,D,S,B,C identik; mekanisme A berbeda. Untuk context variants: B_effective=B_anchor+B_adaptation.",
        "guard": "Jangan mengklaim equal-total-budget architecture superiority.",
    },
    "RA-11": {
        "short": "Studi ini belum melakukan deployment atau field trial. Evidence yang tersedia adalah evaluasi retrospektif controller dalam Virtual Garden dengan forcing meteorologi observasional.",
        "technical": "Belum ada sensor/actuator field integration, crop yield measurement, multi-location soil calibration, safety validation, atau calibrated pump-power model. IoT hanya konteks sensing dan implementasi.",
        "math": "E_field=∅ ⇒ claim_field=NOT_RELEASED.",
        "guard": "External field effectiveness, yield gain, deployment readiness, dan energy savings tidak boleh diklaim.",
    },
    "RA-12": {
        "short": "Hasil nol dan negatif dipertahankan. Forecast main effect dan interaction tidak didukung, sementara memory effect didukung; H2 khusus tiga algoritma tetap inconclusive.",
        "technical": "Main SACSI contrasts memakai exact paired sign-flip, Holm correction, Cohen's dz, dan paired bootstrap CI. Klaim dibuka hanya bila arah efek positif dan p terkoreksi memenuhi aturan; branch activation tetap dipisahkan.",
        "math": "activation ≠ benefit ≠ superiority; release superiority hanya jika Δ>0 dan p_Holm<0.05.",
        "guard": "Null result bukan kegagalan teknis dan tidak boleh disembunyikan atau dibalik menjadi klaim positif.",
    },
}


DEFENSE_QA = (
    ("DQ-01", "mengapa SAC", "Mengapa memilih SAC sebagai anchor?", "SAC sesuai untuk aksi irigasi kontinu dan menyediakan stochastic off-policy learning dengan twin critics serta entropy regularization.", "SAC dipertahankan mekanismenya dan dibandingkan di common environment; ia juga menjadi anchor warm-start bagi varian context.", "J(π)=E[Σγ^t(r_t+αH_t)].", "RA-04"),
    ("DQ-02", "mengapa POMDP", "Mengapa masalah ini diformulasikan sebagai POMDP?", "Policy tidak melihat seluruh state dan proses laten tanah, sementara forcing masa depan tidak diketahui sempurna.", "Current observation, history representation, dan forecast proxy membentuk context yang lebih informatif, tetapi manfaatnya tetap diuji empiris.", "x_(t+1)=f(x_t,a_t,w_t); o_t=h(x_t,w_t).", "RA-03"),
    ("DQ-03", "mengapa DDPG/TD3 sebagai comparator", "Mengapa DDPG dan TD3 dipakai sebagai pembanding?", "Keduanya adalah continuous-control baselines yang memisahkan deterministic single critic dari perbaikan twin critics dan delayed updates.", "Tangga comparator memungkinkan perbandingan sebelum entropy SAC dan context POMDP diperkenalkan.", "TD3 target ∝ min(Q′_1,Q′_2).", "RA-05"),
    ("DQ-04", "fairness benchmark", "Apa yang membuat benchmark ini fair?", "Environment, observation, action, reward, split, matched seeds, interaction budget, metric engine, dan checkpoint selection dikunci.", "Mekanisme intrinsik algoritma tidak disamakan secara artifisial; failed seeds juga tetap dipertahankan.", "CommonConfigHash_DDPG=CommonConfigHash_TD3=CommonConfigHash_SAC.", "RA-10"),
    ("DQ-05", "reward multi-objective", "Mengapa reward harus multi-objective?", "Irigasi harus menjaga kelembapan sambil membatasi air, violation, dan perubahan aktuator.", "reward_v4 dipilih melalui ablation, sensitivity, Pareto trade-off, dan sepuluh-seed confirmation pada validation 2024.", "r_t=2−[100(2d+s)+0.01I+0.01|ΔI|+2V].", "RA-09"),
    ("DQ-06", "raw vs simulated vs synthetic data", "Data mana yang nyata dan mana yang simulasi?", "Meteorologi adalah real/raw, soil state adalah simulated, dan forecast controller adalah controlled synthetic proxy.", "Pemisahan ini berlaku pada seluruh tabel, dashboard, dan narasi disertasi.", "E={M_raw,S_sim,F_proxy}.", "RA-08"),
    ("DQ-07", "virtual-garden scope", "Apakah Virtual Garden merupakan digital twin?", "Tidak. Virtual Garden adalah simulator neraca air hortikultura generik untuk eksperimen terkontrol.", "Ia lulus gate numerik dan simple-case, tetapi belum dikalibrasi sebagai digital twin tanaman, tanah, atau lokasi tertentu.", "|mass-balance residual|≤10⁻⁸ mm sebagai gate numerik.", "RA-07"),
    ("DQ-08", "optimization vs efficiency", "Apa beda optimasi dan efisiensi dalam penelitian ini?", "Optimasi adalah proses belajar terhadap reward; efisiensi adalah outcome fisik yang membaca water bersama Time in Target dan violation.", "DDPG/TD3 pada beberapa seed memakai sedikit air karena under-irrigation, jadi water minimum saja menyesatkan.", "Efficiency assessment ≠ min(I); gunakan Pareto(I, Time-in-Target).", "RA-09"),
    ("DQ-09", "partial observability", "Bagaimana history membantu partial observability?", "LSTM merangkum observation kausal agar policy memiliki representasi dinamika yang tidak tampak dari satu observation.", "Inferensi final mendukung memory main effect, tetapi generalisasi di luar locked pipeline belum diuji.", "z_t=LSTM(o_(t−k:t)).", "RA-03|RA-06"),
    ("DQ-10", "negative result interpretation", "Bagaimana menafsirkan hasil forecast yang tidak signifikan?", "Standalone forecast proxy tidak terbukti meningkatkan primary endpoint pada desain final; itu adalah temuan ilmiah, bukan data yang dibuang.", "Kemungkinan penyebab seperti proxy quality, horizon h+1, dan redundancy dengan current observation dibahas sebagai hipotesis future work, bukan kesimpulan kausal.", "Forecast effect CI melintasi nol dan exact-Holm p≥0.05.", "RA-06|RA-12"),
    ("DQ-11", "statistical evidence", "Apa bukti statistik utama?", "Empat pipeline berbeda pada Friedman omnibus dan ketiga SACSI planned contrasts positif serta lolos exact-Holm.", "Effect size dan paired bootstrap CI dilaporkan; memory effect didukung sedangkan forecast dan interaction tidak.", "Release jika Δ>0 ∧ p_Holm<0.05.", "RA-12"),
    ("DQ-12", "SACSI superiority", "Apakah SACSI terbukti superior?", "Locked SACSI pipeline memiliki Time in Target lebih tinggi daripada SAC, TD3, dan DDPG pada matched-seed retrospective simulation benchmark.", "Klaim ini didukung planned exact sign-flip contrasts, tetapi tidak berarti keunggulan universal atau field effectiveness.", "Δ_SAC=+0.902 pp; Δ_TD3=+13.168 pp; Δ_DDPG=+13.740 pp; semua exact-Holm p=0.046875.", "RA-12"),
    ("DQ-13", "interaction-budget limitation", "Apakah budget SACSI sama dengan model lain?", "Tidak untuk effective total interactions. Context variants menggunakan SAC anchor lalu adaptation stage.", "Main comparators memakai 6,720 total interactions; context variants memakai anchor 6,720 ditambah adaptation 6,720. Karena itu klaim dibatasi pada pipeline.", "B_context=6,720+6,720; B_non-context=6,720.", "RA-10"),
    ("DQ-14", "matched seeds", "Mengapa menggunakan sepuluh matched seeds?", "Matched seeds mengurangi variasi antar-run ketika menghitung selisih policy pada kondisi yang sama.", "Semua failed validation seeds dipertahankan dan perbandingan memakai selisih within-seed, bukan mengganti seed yang buruk.", "d_i=Y_(SACSI,i)−Y_(comparator,i), n=10.", "RA-02|RA-12"),
    ("DQ-15", "deployment readiness", "Apakah sistem siap dipasang di kebun?", "Belum. Hasil menunjukkan simulation evidence, bukan deployment readiness.", "Masih diperlukan sensor/actuator integration, calibration, fail-safe control, archived forecasts, hardware-in-the-loop, dan field trials.", "E_field=∅ ⇒ claim_deployment=NOT_RELEASED.", "RA-11"),
    ("DQ-16", "novelty", "Apa novelty utama jika forecast effect tidak signifikan?", "Novelty utama adalah kerangka evaluasi modular SACSI-POMDP yang mengintegrasikan current state, memory, forecast context, validated simulator, controlled ablation, dan multi-seed inference.", "Novelty integrasi dan metodologi tidak bergantung pada semua komponen harus menghasilkan efek positif; null forecast result justru memperjelas kontribusi memory.", "Novelty framework ≠ claim bahwa setiap component effect >0.", "RA-03|RA-06|RA-12"),
)


CLAIM_LINKS = {
    "C01": ("H1", "RA-07", "DQ-07"),
    "C02": ("H1", "RA-09", "DQ-05"),
    "C03": ("H2", "RA-05|RA-10", "DQ-03|DQ-04"),
    "C04": ("H3", "RA-03|RA-06|RA-12", "DQ-02|DQ-09"),
    "C05": ("H3", "RA-06|RA-12", "DQ-10"),
    "C06": ("H3", "RA-06|RA-12", "DQ-09"),
    "C07": ("H3", "RA-06|RA-12", "DQ-10"),
    "C08": ("H4", "RA-05|RA-10|RA-12", "DQ-11|DQ-12|DQ-13"),
    "C09": ("H3", "RA-06|RA-12", "DQ-10"),
    "C10": ("H1", "RA-08", "DQ-06"),
    "C11": ("N/A", "RA-11", "DQ-15"),
    "C12": ("N/A", "RA-09|RA-11", "DQ-15"),
    "C13": ("N/A", "RA-11", "DQ-15"),
}


def build_defense_package() -> dict:
    registry, reviewer_evidence, dashboard = load_dashboard_release()
    if dashboard.get("status") != "READY":
        raise RuntimeError("Module 9A release is NOT READY")

    dissertation_metadata_path = DISsertation_RESULTS / "dissertation_release_metadata.json"
    dissertation = json.loads(dissertation_metadata_path.read_text(encoding="utf-8"))
    if dissertation.get("status") != "READY":
        raise RuntimeError("Module 9B release is NOT READY")
    for record in dissertation["outputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Module 9B output hash mismatch: {record['path']}")

    source_hash = registry.set_index("evidence_path")["sha256"].to_dict()

    def require_source(path: str) -> str:
        digest = source_hash.get(path)
        if not digest:
            raise RuntimeError(f"Unregistered defense source: {path}")
        return digest

    response_rows = []
    for row in reviewer_evidence.itertuples(index=False):
        answer = REVIEWER_ANSWERS[row.reviewer_item_id]
        sources = row.evidence_file.split("|")
        response_rows.append({
            "reviewer_item_id": row.reviewer_item_id,
            "reviewer_comment": row.reviewer_question_or_input,
            "short_answer_30_60_sec": answer["short"],
            "technical_answer": answer["technical"],
            "mathematical_support": answer["math"],
            "evidence_source": row.evidence_file,
            "evidence_sha256": "|".join(f"{source}={require_source(source)}" for source in sources),
            "dashboard_page": row.dashboard_page,
            "claim_status": row.claim_status,
            "claim_guard": answer["guard"],
            "readiness_status": "READY",
        })
    responses = pd.DataFrame(response_rows)

    source_by_reviewer = responses.set_index("reviewer_item_id")["evidence_source"].to_dict()
    qa_rows = []
    for question_id, topic, question, short, technical, math, reviewer_ids in DEFENSE_QA:
        sources = []
        for reviewer_id in reviewer_ids.split("|"):
            sources.extend(source_by_reviewer[reviewer_id].split("|"))
        sources = list(dict.fromkeys(sources))
        guard = " | ".join(REVIEWER_ANSWERS[item]["guard"] for item in reviewer_ids.split("|"))
        qa_rows.append({
            "question_id": question_id,
            "required_topic": topic,
            "question": question,
            "short_answer_30_60_sec": short,
            "technical_answer": technical,
            "mathematical_support": math,
            "evidence_source": "|".join(sources),
            "evidence_sha256": "|".join(f"{source}={require_source(source)}" for source in sources),
            "claim_guard": guard,
            "readiness_status": "READY",
        })
    qa = pd.DataFrame(qa_rows)

    claims = pd.read_csv(DISsertation_RESULTS / "claim_matrix.csv")
    claim_rows = []
    for row in claims.itertuples(index=False):
        hypothesis, reviewer_ids, question_ids = CLAIM_LINKS[row.claim_id]
        sources = row.evidence_files.split("|")
        claim_rows.append({
            "claim_id": row.claim_id,
            "claim_level": row.claim_level,
            "claim_topic": row.claim_topic,
            "release_status": row.release_status,
            "released_wording": row.released_wording,
            "prohibited_wording": row.prohibited_wording,
            "hypothesis_id": hypothesis,
            "reviewer_item_ids": reviewer_ids,
            "defense_question_ids": question_ids,
            "evidence_files": row.evidence_files,
            "evidence_sha256": "|".join(f"{source}={require_source(source)}" for source in sources),
            "readiness_status": "READY",
        })
    claim_map = pd.DataFrame(claim_rows)

    required_topics = {
        "mengapa SAC", "mengapa POMDP", "mengapa DDPG/TD3 sebagai comparator",
        "fairness benchmark", "reward multi-objective", "raw vs simulated vs synthetic data",
        "virtual-garden scope", "optimization vs efficiency", "partial observability",
        "negative result interpretation", "statistical evidence",
    }
    if (
        set(responses["reviewer_item_id"]) != set(REVIEWER_ANSWERS)
        or len(responses) != 12
        or not required_topics.issubset(set(qa["required_topic"]))
        or set(claim_map["claim_id"]) != set(CLAIM_LINKS)
        or responses["readiness_status"].ne("READY").any()
        or qa["readiness_status"].ne("READY").any()
        or claim_map["readiness_status"].ne("READY").any()
    ):
        raise RuntimeError("Module 9C coverage gate failed")

    red_rows = [
        (row.prohibited_wording, row.released_wording, row.release_status, row.evidence_files)
        for row in claim_map.itertuples(index=False)
    ]
    red_rows.extend((
        ("Forecast meningkatkan performa SACSI.", "Standalone forecast main effect tidak didukung; hanya memory main effect yang didukung.", "BLOCK", "Results/Confirmatory_10Seed/factorial_inference.csv"),
        ("SACSI lebih baik daripada SAC + LSTM.", "Perbandingan SACSI Full dengan SAC + LSTM tidak signifikan setelah exact-Holm.", "BLOCK", "Results/Confirmatory_10Seed/planned_contrasts.csv"),
        ("DDPG, TD3, dan SAC terbukti berbeda signifikan.", "Fair comparison tersedia, tetapi hipotesis inferensial langsung khusus tiga algoritma belum konklusif.", "BLOCK", "Results/Dissertation_Evidence/hypothesis_decision_table.csv"),
        ("Air paling sedikit berarti controller paling efisien.", "Water harus dibaca bersama Time in Target, violation, deficit, dan Pareto trade-off.", "BLOCK", "Results/Confirmatory_10Seed/main_10seed_results_2025.csv"),
        ("Semua seed berhasil melewati validation gate.", "Failed validation seeds dipertahankan sesuai protokol.", "BLOCK", "Results/Confirmatory_10Seed/final_statistics_summary.json"),
    ))

    table_lines = [
        "| Jangan katakan | Gunakan wording | Status | Evidence |",
        "|---|---|---|---|",
    ]
    table_lines.extend(f"| {bad} | {safe} | {status} | `{sources}` |" for bad, safe, status, sources in red_rows)
    red_flags = """# Red-Flag Wording — Module 9C

Gunakan daftar ini saat menulis, presentasi, dan menjawab reviewer. Jika wording pada kolom pertama muncul, ganti dengan wording yang dibatasi evidence.

""" + "\n".join(table_lines) + """

## Aturan lisan cepat

1. Awali klaim performa dengan **“under the locked warm-start training pipelines”**.
2. Sebut evaluasi sebagai **retrospective 2025 Virtual Garden simulation benchmark**.
3. Sebut forecast sebagai **SF-20 h+1 controlled synthetic forecast proxy**.
4. Pisahkan **framework validity**, **context activation**, **performance benefit**, dan **statistical superiority**.
5. Jika ditanya deployment, jawab **belum diuji di lapangan**.
"""

    main = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "main_10seed_results_2025.csv")
    planned = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "planned_contrasts.csv")
    effects = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "factorial_inference.csv")
    main_means = main.groupby("model")["time_in_target_pct"].mean()
    main_planned = planned.loc[planned["analysis_family"].eq("main_benchmark")]
    memory = effects.loc[effects["effect"].eq("Memory main effect")].iloc[0]
    defense_card = f"""# One-Page Defense Card — SACSI-POMDP

## Satu kalimat kontribusi

SACSI-POMDP adalah kerangka evaluasi modular untuk continuous smart-irrigation control yang menggabungkan current observation, temporal memory, dan controlled forecast context dalam Virtual Garden, dengan ablation terkontrol dan confirmatory matched-seed inference.

## Desain yang harus diingat

- Scope: hortikultura generik dalam Virtual Garden; belum ada field validation.
- Data: real/raw meteorology → simulated soil state → SF-20 h+1 controlled synthetic forecast proxy.
- Split: training 2021–2023; reward/checkpoint selection 2024; retrospective benchmark 2025.
- Primary endpoint: Time in Target (%).
- Main comparison: DDPG, TD3, SAC, dan SACSI-POMDP pada 10 matched seeds.
- Factorial: SAC Basic, SAC + Forecast, SAC + LSTM, SACSI Full.

## Angka inti

- Mean Time in Target: SACSI {main_means['SACSI-POMDP']:.3f}%; SAC {main_means['SAC']:.3f}%; TD3 {main_means['TD3']:.3f}%; DDPG {main_means['DDPG']:.3f}%.
- SACSI differences: SAC +{main_planned.iloc[0]['mean_difference_pp']:.3f} pp; TD3 +{main_planned.iloc[1]['mean_difference_pp']:.3f} pp; DDPG +{main_planned.iloc[2]['mean_difference_pp']:.3f} pp; seluruh exact-Holm p = {main_planned.iloc[0]['primary_p_holm']:.6f}.
- Memory main effect: +{memory['mean_effect']:.3f} pp, 95% bootstrap CI [{memory['bootstrap_ci95_low']:.3f}, {memory['bootstrap_ci95_high']:.3f}], exact-Holm p = {memory['exact_sign_flip_p_holm']:.6f}.
- Forecast main effect dan forecast × memory interaction: tidak didukung.

## Keputusan H1–H4

- H1: supported engineering gates; bukan field validation.
- H2: direct three-algorithm inference inconclusive; laporkan hasil deskriptif saja.
- H3: reject null dengan component guard; memory didukung, forecast dan interaction null.
- H4: reject null untuk locked pipeline scope; bukan equal-total-budget architecture claim.

## Jawaban aman saat ditekan reviewer

- “Superior?” → **Ya untuk locked pipeline dan Time in Target pada retrospective simulation benchmark; bukan universal atau field claim.**
- “Forecast berguna?” → **Branch aktif, tetapi standalone performance benefit tidak didukung.**
- “Mengapa tetap novel?” → **Novelty berada pada integrasi modular, controlled ablation, dan evidence hierarchy; novelty tidak mensyaratkan semua efek positif.**
- “Fair budget?” → **Main DDPG/TD3/SAC fair; SACSI context pipeline memiliki tambahan adaptation budget, sehingga klaim arsitektur equal-budget ditahan.**
- “Siap deploy?” → **Belum; diperlukan calibration, hardware-in-the-loop, archived forecasts, dan field trials.**

## Source cepat

`Results/Confirmatory_10Seed/main_10seed_results_2025.csv` · `planned_contrasts.csv` · `factorial_inference.csv` · `final_statistics_summary.json` · `Results/Reviewer_Defense/claim_to_evidence_matrix.csv`
"""

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "reviewer_response_matrix.csv": RESULTS / "reviewer_response_matrix.csv",
        "defense_qa_bank.csv": RESULTS / "defense_qa_bank.csv",
        "claim_to_evidence_matrix.csv": RESULTS / "claim_to_evidence_matrix.csv",
        "red_flag_wording.md": DOCS / "red_flag_wording.md",
        "one_page_defense_card.md": DOCS / "one_page_defense_card.md",
    }
    responses.to_csv(output_paths["reviewer_response_matrix.csv"], index=False)
    qa.to_csv(output_paths["defense_qa_bank.csv"], index=False)
    claim_map.to_csv(output_paths["claim_to_evidence_matrix.csv"], index=False)
    output_paths["red_flag_wording.md"].write_text(red_flags, encoding="utf-8")
    output_paths["one_page_defense_card.md"].write_text(defense_card, encoding="utf-8")

    output_records = {
        name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for name, path in output_paths.items()
    }
    metadata_path = RESULTS / "defense_release_metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata = {
        "module": "9C",
        "status": "READY",
        "dashboard_release_status": dashboard["status"],
        "dissertation_release_status": dissertation["status"],
        "reviewer_items_answered": len(responses),
        "reviewer_coverage_pct": float(len(responses) / len(REVIEWER_ANSWERS) * 100),
        "defense_questions": len(qa),
        "mandatory_topics_covered": len(required_topics),
        "claim_rows": len(claim_map),
        "unsupported_claim_warnings": len(red_rows),
        "outputs": output_records,
        "generated_utc": previous.get("generated_utc") if previous.get("outputs") == output_records else datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    release = build_defense_package()
    print(json.dumps(release, indent=2))
    if release["status"] != "READY":
        raise SystemExit("Defense package is NOT READY")
