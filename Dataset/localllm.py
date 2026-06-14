
import os

import requests


OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:16434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))


def generate_with_local_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = OLLAMA_GENERATE_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def build_mock_cv_prompt(title, jd_context):
    return f"""Bạn là một chuyên gia nhân sự. Hãy tạo danh sách 20 CV tiếng Việt giả định cho nhóm vị trí liên quan đến: '{title}'. Không cần nhất thiết phải trùng khớp với {title}
{jd_context}

VÍ DỤ CV ĐÚNG CHUẨN:
[
  {{
    "Tên ứng viên": "Nguyễn Minh Anh",
    "Vị trí ứng tuyển": "Chuyên viên Phân tích Dữ liệu",
    "Lĩnh vực": "Công nghệ thông tin",
    "Mục tiêu nghề nghiệp": "Tôi mong muốn ứng dụng kỹ năng phân tích dữ liệu, SQL và trực quan hóa báo cáo để hỗ trợ doanh nghiệp ra quyết định chính xác hơn, đồng thời phát triển chuyên môn về phân tích hành vi khách hàng và tối ưu hiệu quả vận hành.",
    "Kinh nghiệm": "2 năm phân tích dữ liệu bán hàng, xây dựng dashboard Power BI, xử lý dữ liệu bằng SQL và Python.",
    "Nơi làm việc mong muốn": "TP. Hồ Chí Minh",
    "Kỹ năng": "SQL, Python, Power BI, Excel nâng cao, phân tích dữ liệu, giao tiếp với phòng kinh doanh",
    "Bằng cấp": "Cử nhân Hệ thống thông tin quản lý - Đại học Kinh tế TP. Hồ Chí Minh",
    "Tình trạng hôn nhân": "Độc thân",
    "Mức lương mong muốn": "18-22 triệu VND",
    "Giới tính": "Nữ",
    "Tuổi": 26
  }},
  {{
    "Tên ứng viên": "Trần Quốc Huy",
    "Vị trí ứng tuyển": "Kỹ sư Backend Java",
    "Lĩnh vực": "Phát triển phần mềm",
    "Mục tiêu nghề nghiệp": "Tôi định hướng phát triển thành kỹ sư backend vững chuyên môn, xây dựng hệ thống ổn định, bảo mật và dễ mở rộng. Tôi muốn đóng góp vào các sản phẩm có lượng người dùng lớn và học sâu hơn về kiến trúc microservices.",
    "Kinh nghiệm": "3 năm phát triển API bằng Java Spring Boot, tích hợp PostgreSQL, Redis, Docker và triển khai CI/CD.",
    "Nơi làm việc mong muốn": "Hà Nội",
    "Kỹ năng": "Java, Spring Boot, REST API, PostgreSQL, Redis, Docker, Git, microservices",
    "Bằng cấp": "Cử nhân Kỹ thuật phần mềm - Đại học Công nghệ",
    "Tình trạng hôn nhân": "Đã kết hôn",
    "Mức lương mong muốn": "28-35 triệu VND",
    "Giới tính": "Nam",
    "Tuổi": 29
  }},
  {{
    "Tên ứng viên": "Lê Thảo Vy",
    "Vị trí ứng tuyển": "Chuyên viên Marketing Nội dung",
    "Lĩnh vực": "Marketing",
    "Mục tiêu nghề nghiệp": "Tôi mong muốn phát triển các chiến dịch nội dung có tính thuyết phục, phù hợp hành vi khách hàng và mục tiêu thương hiệu. Tôi hướng tới vai trò quản lý nội dung, kết hợp sáng tạo với đo lường hiệu quả truyền thông.",
    "Kinh nghiệm": "2 năm viết nội dung website, quản lý fanpage, phối hợp chạy chiến dịch email marketing và đo lường hiệu quả bằng Google Analytics.",
    "Nơi làm việc mong muốn": "Đà Nẵng",
    "Kỹ năng": "Content marketing, SEO, Google Analytics, Facebook Business Suite, lập kế hoạch nội dung, viết quảng cáo",
    "Bằng cấp": "Cử nhân Truyền thông Marketing - Đại học Kinh tế Đà Nẵng",
    "Tình trạng hôn nhân": "Độc thân",
    "Mức lương mong muốn": "14-18 triệu VND",
    "Giới tính": "Nữ",
    "Tuổi": 25
  }},
  {{
    "Tên ứng viên": "Phạm Đức Long",
    "Vị trí ứng tuyển": "Kế toán Tổng hợp",
    "Lĩnh vực": "Tài chính - Kế toán",
    "Mục tiêu nghề nghiệp": "Tôi muốn vận dụng kiến thức kế toán, thuế và kiểm soát chứng từ để đảm bảo số liệu tài chính chính xác, đúng quy định. Mục tiêu của tôi là phát triển lên vị trí kế toán trưởng trong môi trường chuyên nghiệp.",
    "Kinh nghiệm": "4 năm xử lý sổ sách kế toán, lập báo cáo thuế, đối chiếu công nợ, kiểm tra chứng từ và làm việc với kiểm toán.",
    "Nơi làm việc mong muốn": "Bình Dương",
    "Kỹ năng": "Kế toán tổng hợp, báo cáo thuế, MISA, Excel, đối chiếu công nợ, kiểm soát chứng từ",
    "Bằng cấp": "Cử nhân Kế toán - Đại học Tài chính Marketing",
    "Tình trạng hôn nhân": "Đã kết hôn",
    "Mức lương mong muốn": "20-25 triệu VND",
    "Giới tính": "Nam",
    "Tuổi": 31
  }},
  {{
    "Tên ứng viên": "Hoàng Gia Bảo",
    "Vị trí ứng tuyển": "Nhân viên Kinh doanh B2B",
    "Lĩnh vực": "Kinh doanh",
    "Mục tiêu nghề nghiệp": "Tôi mong muốn phát triển năng lực bán hàng B2B, mở rộng mạng lưới khách hàng doanh nghiệp và đạt chỉ tiêu doanh số bền vững. Tôi hướng tới vai trò trưởng nhóm kinh doanh thông qua kỹ năng tư vấn và chăm sóc khách hàng.",
    "Kinh nghiệm": "3 năm tư vấn giải pháp cho khách hàng doanh nghiệp, quản lý pipeline CRM, đàm phán hợp đồng và chăm sóc khách hàng sau bán.",
    "Nơi làm việc mong muốn": "TP. Hồ Chí Minh",
    "Kỹ năng": "Bán hàng B2B, đàm phán, CRM, tư vấn giải pháp, chăm sóc khách hàng, thuyết trình",
    "Bằng cấp": "Cử nhân Quản trị kinh doanh - Đại học Mở TP. Hồ Chí Minh",
    "Tình trạng hôn nhân": "Độc thân",
    "Mức lương mong muốn": "16-22 triệu VND + hoa hồng",
    "Giới tính": "Nam",
    "Tuổi": 27
  }}
]

YÊU CẦU:
- Tạo ra 20 CV hoàn toàn khác nhau, không trùng lặp.
- Vị trí ứng tuyển của cả 20 CV không cần trùng chính xác với '{title}', nhưng phải liên quan chặt chẽ đến '{title}' và phù hợp với JD.
- Có thể dùng các biến thể chức danh hợp lý như junior/senior, chuyên viên, kỹ sư, thực tập sinh, trưởng nhóm hoặc các vai trò gần với '{title}'.
- Mục tiêu nghề nghiệp dài khoảng 30-50 từ và phù hợp với thông tin công việc thực tế.
- Kỹ năng, kinh nghiệm và bằng cấp phải bám sát yêu cầu công việc.
- Nếu ứng viên có bằng Thạc sĩ, liệt kê cả bằng Cử nhân trước đó.
- Tình trạng hôn nhân chọn ngẫu nhiên "Độc thân" hoặc "Đã kết hôn".
- Chỉ trả về đúng một JSON Array, không markdown, không giải thích.
- Mỗi object có đúng các key:
  "Tên ứng viên", "Vị trí ứng tuyển", "Lĩnh vực", "Mục tiêu nghề nghiệp",
  "Kinh nghiệm", "Nơi làm việc mong muốn", "Kỹ năng", "Bằng cấp",
  "Tình trạng hôn nhân", "Mức lương mong muốn", "Giới tính", "Tuổi".
"""


def build_enrich_cv_prompt(
    name,
    job,
    industry,
    workplace,
    marriage,
    salary,
    gender,
    age,
):
    return f"""Bạn là chuyên gia nhân sự. Đây là thông tin của 1 ứng viên có thật nhưng còn thiếu nhiều phần:

Tên: {name}
Vị trí: {job}
Lĩnh vực: {industry}
Nơi làm việc: {workplace}
Tình trạng hôn nhân: {marriage}
Mức lương mong muốn: {salary}
Giới tính: {gender}
Tuổi: {age}

NHIỆM VỤ:
1. Viết "Mục tiêu nghề nghiệp" ngắn gọn, khoảng 30-50 từ, phù hợp với vị trí và lĩnh vực.
2. Tạo "Kinh nghiệm", "Kỹ năng", và "Bằng cấp" hợp lý với vị trí.
3. Nếu "Nơi làm việc mong muốn" trống, hãy điền một địa điểm ở Việt Nam.
4. Giữ nguyên Tên, Vị trí, Lĩnh vực, và Tình trạng hôn nhân.
5. Chỉ trả về đúng một JSON object hợp lệ, không markdown, không giải thích.
6. JSON phải có đúng các key:
   "Tên ứng viên", "Vị trí ứng tuyển", "Lĩnh vực", "Mục tiêu nghề nghiệp",
   "Kinh nghiệm", "Nơi làm việc mong muốn", "Kỹ năng", "Bằng cấp",
   "Tình trạng hôn nhân", "Mức lương mong muốn", "Giới tính", "Tuổi".
"""
