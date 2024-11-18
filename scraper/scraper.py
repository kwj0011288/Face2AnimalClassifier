from simple_image_download import simple_image_download as simp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize the simple_image_download class
my_downloader = simp.simple_image_download

# Set the desired file format extension
my_downloader.extensions = '.jpg'

# Define categories and keywords
categories = {
    "dog": ['블랙핑크 지수', '체리블렛 유주', '아이즈원 안유진', '프로미스나인 장규리', '여자아이들 우기', '위클리 이수진', '에스파 윈터', '위키미키 지수연', '로켓펀치 소희', '프로미스나인 백지헌', '송중기', '박보검', '박보영', '워너원 강다니엘', '엑소 백현'],
    "cat": ['블랙핑크 제니', '드림캐쳐 지유', '씨엘씨 장예은', '있지 류진', '있지 예나', '엘리스 소희', '로켓펀치 연희', '우아! 나나', '한예슬', '안소희', '김현아', '이효리', '워너원 황민현', '엑소 시우민', '강동원', '이종석', '이준기'],
    "rabbit": ['트와이스 나연', '씨엘씨 최유진', '오마이걸 효정', '프로미스나인 박지원', '스테이씨 수민', '엘리스 유경', '우주소녀 보나', '이달의 소녀 희진', '아이즈원 장원영', '있지 유나', '수지', '아이유', '방탄소년단 정국', '아이콘 바비', '워너원 박지훈', '엑소 수호'],
    "turtle": ['마마무 솔사', '레드벨벳 예리', '브레이브걸스 유정', '엘리스 벨라', '에버글로우 온다', '우아! 민서', '피에스타 김재이', '김태리', 'skadbwjd', '샤이니 민호', '세븐틵 정한', '주결경', '하연수', '허훈', '몬스터엑스 형원', 'HKT48 후치가미 마이'],
    "bear": ['마동석', '조진웅', '조세호', '김대명', '김해준', '더원', '셔누', '스즈키 료헤이', '안재홍', '안창림', '유지태', '윤균상', '이상윤', '윤민수', '최자']
}

# Iterate over each category and download images
for category, keywords_list in categories.items():
    for keyword in keywords_list:  # Download images for each keyword separately
        my_downloader().download(keyword, limit=800)
        print(f"Images downloaded for {keyword} in category {category}")
