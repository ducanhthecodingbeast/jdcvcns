import json
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CV_KEYS = {
    "name": ("Tên ứng viên", "TÃªn á»©ng viÃªn"),
    "target": ("Vị trí ứng tuyển", "Vá»‹ trÃ­ á»©ng tuyá»ƒn"),
    "industry": ("Lĩnh vực", "LÄ©nh vá»±c"),
    "objective": ("Mục tiêu nghề nghiệp", "Má»¥c tiÃªu nghá» nghiá»‡p", "Mục tiêu", "Má»¥c tiÃªu"),
    "experience": ("Kinh nghiệm", "Kinh nghiá»‡m"),
    "location": ("Nơi làm việc mong muốn", "NÆ¡i lÃ m viá»‡c mong muá»‘n"),
    "skills": ("Kỹ năng", "Ká»¹ nÄƒng"),
    "education": ("Bằng cấp", "Báº±ng cáº¥p"),
    "marriage": ("Tình trạng hôn nhân", "TÃ¬nh tráº¡ng hÃ´n nhÃ¢n"),
    "salary": ("Mức lương mong muốn", "Má»©c lÆ°Æ¡ng mong muá»‘n"),
    "gender": ("Giới tính", "Giá»›i tÃ­nh"),
    "age": ("Tuổi", "Tuá»•i"),
}

JD_KEYS = {
    "title": ("Vị trí cần tuyển", "Vá»‹ trÃ­ cáº§n tuyá»ƒn", "Job Title"),
    "company": ("Tên công ty", "TÃªn cÃ´ng ty", "Name Company"),
    "size": ("Quy mô công ty", "Quy mÃ´ cÃ´ng ty"),
    "address": ("Địa chỉ công ty", "Äá»‹a chá»‰ cÃ´ng ty"),
    "overview": ("Giới thiệu công ty", "Giá»›i thiá»‡u cÃ´ng ty"),
    "description": ("Mô tả công việc", "MÃ´ táº£ cÃ´ng viá»‡c"),
    "requirements": ("Yêu cầu công việc", "YÃªu cáº§u cÃ´ng viá»‡c"),
    "benefits": ("Quyền lợi", "Quyá»n lá»£i"),
}


def value(row, keys: Iterable[str], default="N/A"):
    for key in keys:
        if key in row and pd.notna(row.get(key)):
            return str(row.get(key))
    return default


def build_recommendations(score_matrix, top_k=10):
    recommendations = []
    for scores in score_matrix:
        top_idx = np.argsort(scores)[::-1][:top_k]
        recommendations.append([(int(idx), float(scores[int(idx)])) for idx in top_idx])
    return recommendations


def build_two_stage_hybrid_recommendations(v2_scores, dense_scores, lexical_scores, retrieve_k=30, top_k=10):
    recommendations = []
    for i, first_stage_scores in enumerate(v2_scores):
        top_retrieved = np.argsort(first_stage_scores)[::-1][:retrieve_k]
        ranked = [
            (int(jd_idx), float(0.5 * dense_scores[i][jd_idx] + 0.5 * lexical_scores[i][jd_idx]))
            for jd_idx in top_retrieved
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        recommendations.append(ranked[:top_k])
    return recommendations


def _page(title, subtitle, body, modal_json=None):
    modal_script = ""
    if modal_json is not None:
        modal_script = f"""
  <div class="modal-overlay" id="detail-modal" onclick="if(event.target === this) closeModal()">
    <div class="modal-box">
      <div class="modal-header">
        <div>
          <h2 id="modal-title">Detail</h2>
          <p id="modal-subtitle"></p>
        </div>
        <button class="modal-close" onclick="closeModal()">×</button>
      </div>
      <pre class="modal-body" id="modal-body"></pre>
    </div>
  </div>
  <script>
    const modalData = {json.dumps(modal_json, ensure_ascii=False)};
    function openModal(id) {{
      const item = modalData[id];
      if (!item) return;
      document.getElementById('modal-title').innerText = item.title || 'Detail';
      document.getElementById('modal-subtitle').innerText = item.subtitle || '';
      document.getElementById('modal-body').innerText = item.body || '';
      document.getElementById('detail-modal').classList.add('active');
    }}
    function closeModal() {{
      document.getElementById('detail-modal').classList.remove('active');
    }}
    function filterCards() {{
      const query = document.getElementById('search-input').value.toLowerCase().trim();
      document.querySelectorAll('[data-search]').forEach((card) => {{
        card.style.display = card.getAttribute('data-search').includes(query) ? '' : 'none';
      }});
    }}
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape') closeModal();
    }});
  </script>"""

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>{escape(title)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    body {{ font-family: 'IBM Plex Sans', sans-serif; background: #ffffff; color: #161616; margin: 0; line-height: 1.5; letter-spacing: 0.16px; }}
    .header {{ background: #161616; color: #ffffff; padding: 40px 32px; }}
    .header h1 {{ font-weight: 300; font-size: 40px; margin: 0 0 8px; }}
    .header p {{ color: #c6c6c6; margin: 0; }}
    .container {{ max-width: 1500px; margin: 0 auto; padding: 32px; }}
    .search-input {{ width: min(640px, 100%); padding: 12px 16px; border: 0; border-bottom: 1px solid #8d8d8d; background: #f4f4f4; font: inherit; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
    .card {{ border: 1px solid #e0e0e0; background: #ffffff; padding: 20px; }}
    .card:hover {{ border-color: #0f62fe; background: #f4f4f4; }}
    .title {{ font-size: 20px; font-weight: 400; margin: 0 0 4px; }}
    .meta {{ color: #525252; font-size: 14px; margin-bottom: 12px; }}
    .details {{ border: 1px solid #e0e0e0; background: #ffffff; padding: 12px; margin: 12px 0; font-size: 14px; }}
    .match {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 10px 0; border-top: 1px solid #e0e0e0; }}
    .score {{ color: #0f62fe; font-weight: 600; white-space: nowrap; }}
    button {{ background: #0f62fe; color: #ffffff; border: 0; padding: 8px 12px; cursor: pointer; font: inherit; }}
    .modal-overlay {{ position: fixed; inset: 0; background: rgba(22,22,22,.55); display: none; align-items: center; justify-content: center; padding: 24px; }}
    .modal-overlay.active {{ display: flex; }}
    .modal-box {{ background: #ffffff; border: 1px solid #161616; width: min(760px, 100%); max-height: 88vh; display: flex; flex-direction: column; }}
    .modal-header {{ display: flex; justify-content: space-between; gap: 16px; padding: 20px 24px; border-bottom: 1px solid #e0e0e0; background: #f4f4f4; }}
    .modal-header h2 {{ margin: 0; font-size: 22px; font-weight: 400; }}
    .modal-header p {{ margin: 4px 0 0; color: #525252; }}
    .modal-close {{ background: transparent; color: #161616; font-size: 24px; padding: 0 4px; }}
    .modal-body {{ margin: 0; padding: 24px; overflow: auto; white-space: pre-wrap; font-family: inherit; }}
  </style>
</head>
<body>
  <div class="header"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>
  <div class="container">
    <input type="text" id="search-input" class="search-input" oninput="filterCards()" placeholder="Tìm kiếm...">
    {body}
  </div>
  {modal_script}
</body>
</html>"""


def render_cv_to_jd_report(output_path, df_cv, df_jd, recommendations, title, subtitle):
    modal_data = {}
    cards = ['<div class="grid">']
    for cv_idx, (_, cv_row) in enumerate(df_cv.iterrows()):
        cv_name = value(cv_row, CV_KEYS["name"], f"CV {cv_idx + 1}")
        cv_target = value(cv_row, CV_KEYS["target"])
        search = escape(" ".join(str(v) for v in cv_row.to_dict().values()).lower(), quote=True)
        cards.append(f'<div class="card" data-search="{search}">')
        cards.append(f'<h2 class="title">#{cv_idx + 1} {escape(cv_name)}</h2>')
        cards.append(f'<div class="meta">{escape(cv_target)}</div>')
        cards.append('<div class="details">')
        cards.append(f'<div><strong>Kỹ năng:</strong> {escape(value(cv_row, CV_KEYS["skills"]))}</div>')
        cards.append(f'<div><strong>Kinh nghiệm:</strong> {escape(value(cv_row, CV_KEYS["experience"]))}</div>')
        cards.append("</div>")
        for rank, (jd_idx, score) in enumerate(recommendations[cv_idx], start=1):
            jd_row = df_jd.iloc[int(jd_idx)]
            jd_title = value(jd_row, JD_KEYS["title"], f"JD {jd_idx}")
            jd_company = value(jd_row, JD_KEYS["company"])
            modal_id = f"jd-{cv_idx}-{jd_idx}"
            modal_data[modal_id] = {
                "title": jd_title,
                "subtitle": jd_company,
                "body": "\n\n".join(f"{k}: {v}" for k, v in jd_row.to_dict().items()),
            }
            cards.append('<div class="match">')
            cards.append(f'<div><strong>#{rank} {escape(jd_title)}</strong><br><span class="meta">{escape(jd_company)}</span></div>')
            cards.append(f'<div><span class="score">{score:.4f}</span> <button onclick="openModal(\'{modal_id}\')">Chi tiết</button></div>')
            cards.append("</div>")
        cards.append("</div>")
    cards.append("</div>")
    Path(output_path).write_text(_page(title, subtitle, "\n".join(cards), modal_data), encoding="utf-8")


def render_jd_to_cv_report(output_path, df_jd, df_cv, similarities, title_col=None):
    modal_data = {}
    cards = []
    for jd_idx, (_, jd_row) in enumerate(df_jd.iterrows()):
        jd_title = value(jd_row, (title_col,) if title_col else JD_KEYS["title"], f"JD {jd_idx + 1}")
        jd_company = value(jd_row, JD_KEYS["company"])
        cards.append('<div class="card" data-search="{}">'.format(escape(" ".join(str(v) for v in jd_row.to_dict().values()).lower(), quote=True)))
        cards.append(f'<h2 class="title">#{jd_idx + 1} {escape(jd_title)}</h2>')
        cards.append(f'<div class="meta">{escape(jd_company)} · Top 10 CV matches</div>')
        scores = similarities[jd_idx]
        top_cv_idx = np.argsort(scores)[::-1][:10]
        for rank, cv_idx in enumerate(top_cv_idx, start=1):
            cv_row = df_cv.iloc[int(cv_idx)]
            cv_name = value(cv_row, CV_KEYS["name"], f"CV {cv_idx}")
            cv_target = value(cv_row, CV_KEYS["target"])
            modal_id = f"cv-{jd_idx}-{cv_idx}"
            modal_data[modal_id] = {
                "title": cv_name,
                "subtitle": cv_target,
                "body": "\n\n".join(f"{k}: {v}" for k, v in cv_row.to_dict().items()),
            }
            cards.append('<div class="match">')
            cards.append(f'<div><strong>#{rank} {escape(cv_name)}</strong><br><span class="meta">{escape(cv_target)}</span></div>')
            cards.append(f'<div><span class="score">{float(scores[int(cv_idx)]):.4f}</span> <button onclick="openModal(\'{modal_id}\')">Chi tiết</button></div>')
            cards.append("</div>")
        cards.append("</div>")
    body = '<div class="grid">' + "\n".join(cards) + "</div>"
    Path(output_path).write_text(
        _page("Job Matching Analysis", "JD-to-CV matching report", body, modal_data),
        encoding="utf-8",
    )
