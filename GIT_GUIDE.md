# 🚀 VSCode Git & GitHub 협업 가이드 (팀원 필독)

Git과 GitHub를 처음 사용하는 팀원들을 위한 기초 협업 가이드입니다. **"코드가 꼬이거나 날아가는 일"**을 방지하기 위해 이 문서의 순서를 반드시 지켜주세요!

---

## 🛠 0. VSCode Git 환경 초기 세팅 (최초 1회만)

1. **Git 설치**: [Git 공식 홈페이지](https://git-scm.com/)에서 다운로드 후 설치 (기본 설정으로 Next 계속 클릭).
2. **VSCode 실행**: 왼쪽 메뉴 중 3번째 아이콘인 **'소스 제어(Source Control)'** 아이콘이 정상적으로 보이면 성공.
3. **Git 유저 정보 등록**: VSCode에서 터미널(`Ctrl` + `~`)을 열고 아래 명령어를 입력하세요.
   ```bash
   git config --global user.name "본인영문이름"
   git config --global user.email "본인깃허브이메일@example.com"
   ```

### 🌟 0.1. (강력 추천) VSCode Git 확장 프로그램 설치
VSCode 좌측 5번째 블록 아이콘(Extensions)에서 다음 확장을 검색해 설치하면 코드가 꼬일 확률이 획기적으로 줄어듭니다!
- **Git Graph** (필수👍): 현재 브랜치가 어떻게 갈라지고 합쳐졌는지 예쁜 시각적 그래프로 볼 수 있습니다. 터미널 명령어를 몰라도 화면을 보며 직관적으로 작업할 수 있습니다.
- **GitLens — Git supercharged** (권장): 코드 줄마다 우측에 "누가, 언제, 어떤 커밋으로" 해당 줄을 수정했는지(git blame) 희미한 글씨로 띄워줍니다. 팀원이 짠 코드를 이해할 때 유용합니다.
- **GitHub Pull Requests and Issues**: VSCode 창을 벗어나지 않고도, 에디터 안에서 PR을 확인하고 코드 리뷰를 남길 수 있습니다.

---

## 📥 1. 프로젝트 가져오기 (Clone) - (최초 1회만)

원격 저장소(GitHub)의 코드를 내 컴퓨터로 복사해오는 과정입니다.

1. VSCode 터미널을 열고 코드를 저장할 폴더로 이동합니다.
2. 아래 명령어를 입력합니다.
   ```bash
   git clone https://github.com/msjoon0811/real-estateml.git
   cd real-estateml
   ```
3. 클론이 완료되면 터미널에 아래 명령어를 입력하여 기본 작업 브랜치인 `develop`으로 이동합니다.
   ```bash
   git checkout develop
   ```
4. `.env.example` 파일을 복사하여 `.env` 파일을 만들고 각자 API 키를 넣습니다.

---

## 🌿 2. 내 작업 공간 만들기 (Branch 생성) - ⭐️ 매우 중요

**절대 `main`이나 `develop` 브랜치에서 직접 코드를 수정하지 마세요!** 작업 전 반드시 본인의 기능 브랜치를 만들어야 합니다.

1. 터미널에서 현재 브랜치가 `develop`인지 확인합니다. (`git branch` 입력)
2. 새로운 브랜치를 생성하고 이동합니다. (이름 규칙: `feature/이름-기능`)
   ```bash
   git checkout -b feature/woohyun-api-crawler
   ```
   *(예: `feature/jongin-kobert`, `feature/woohyun-fastapi` 등)*
3. 이제 코드를 마음껏 수정하고 작업하시면 됩니다!

---

## 💾 3. 작업 내역 저장하기 (Add & Commit)

코드를 작성/수정했다면 내역을 로컬에 저장해야 합니다.

1. VSCode 왼쪽의 **소스 제어(Source Control)** 탭(가지 모양 아이콘)을 클릭합니다.
2. '변경 사항(Changes)' 목록에 수정된 파일들이 뜹니다.
3. 파일명 옆의 **`+` 버튼(스테이징)**을 눌러 '스테이징된 변경 사항'으로 올립니다. (`git add`)
4. 위쪽 메시지 입력칸에 **무엇을 변경했는지** 적습니다. (예: `feat: 네이버 뉴스 API 수집 모듈 추가`)
5. **`커밋(Commit)`** 버튼을 누릅니다.

---

## 🚀 4. GitHub에 올리기 (Push) & 합치기 요청 (Pull Request)

내 컴퓨터(로컬)에 커밋한 내용을 GitHub에 올리고, 팀장에게 코드를 합쳐달라고 요청하는 과정입니다.

1. 터미널에 아래 명령어를 입력하여 내 브랜치를 GitHub에 올립니다.
   ```bash
   git push origin 본인브랜치명
   # 예: git push origin feature/woohyun-api-crawler
   ```
2. **GitHub 사이트**에 접속합니다.
3. 상단에 노란색/초록색으로 **"Compare & pull request"** 버튼이 뜹니다. 클릭!
4. 제목과 본문에 **어떤 작업을 했는지** 적고 **"Create pull request"**를 누릅니다.
5. ⚠️ 이제 **팀장(승준)이 코드를 리뷰하고 `develop` 브랜치에 합쳐줄 것**입니다. 본인이 직접 Merge하지 마세요!

---

## 🔄 5. 최신 코드 가져오기 (Pull) - 매일 작업 시작 전!

다른 팀원이 작성해서 `develop`에 합쳐진 최신 코드를 내 컴퓨터로 가져오는 과정입니다. **작업을 시작하기 전 매일 아침 반드시 수행하세요.**

1. 로컬의 `develop` 브랜치로 이동합니다.
   ```bash
   git checkout develop
   ```
2. 최신 코드를 당겨옵니다.
   ```bash
   git pull origin develop
   ```
3. 다시 내 작업 브랜치로 이동하거나, 새로운 기능 작업을 위해 새 브랜치를 만듭니다(`git checkout -b 새로운브랜치`).

---

## 🚨 코드가 꼬이는 것을 막는 핵심 철칙 (팀장 당부)

1. **내 브랜치인지 항상 확인**: VSCode 좌측 하단에 현재 브랜치 이름이 나옵니다. `develop`이나 `main` 상태에서 코딩하지 마세요.
2. **작은 단위로 자주 Commit & Push**: 코드를 1000줄 짜놓고 한 번에 올리면 오류 찾기가 지옥이 됩니다. "뉴스 API 수집 함수 1개 완성" 수준에서 자주 커밋하세요.
3. **Pull은 자주 할수록 좋다**: 다른 사람의 코드 변경사항을 자주 받아와야 내 코드와 나중에 충돌(Conflict)이 안 납니다.
4. **에러가 나면 멈추기**: 터미널에 빨간색 영어가 뜨거나 충돌(Merge Conflict)이 났는데 뭔지 모르겠다면, **절대 강제로 뭘 하려 하지 말고 그대로 둔 상태로 팀장(승준)에게 화면을 보여주세요.**
