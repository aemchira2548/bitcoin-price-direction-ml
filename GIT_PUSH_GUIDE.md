# วิธี Push โปรเจคขึ้น GitHub

## 1) สร้าง repo บน GitHub ก่อน
ไปที่ github.com → New repository → ตั้งชื่อ เช่น `bitcoin-price-direction-ml`
**ไม่ต้อง** ติ๊ก "Add README" (เรามีอยู่แล้ว) → กด Create repository

## 2) เปิด terminal ในโฟลเดอร์โปรเจค แล้วรันทีละคำสั่ง

```bash
cd path/to/project        # เข้าไปในโฟลเดอร์ project ที่มี app.py, src/, data/ ฯลฯ

git init                                   # เริ่มต้น git repo (ทำครั้งเดียว)
git add .                                  # เพิ่มไฟล์ทั้งหมด
git commit -m "Initial commit: Bitcoin ML prediction project"

git branch -M main                         # ตั้งชื่อ branch หลักเป็น main
git remote add origin https://github.com/<username>/bitcoin-price-direction-ml.git
git push -u origin main                    # push ขึ้น GitHub ครั้งแรก
```

แทน `<username>` ด้วยชื่อบัญชี GitHub ของตัวเอง

## 3) ครั้งต่อไปที่แก้ไขไฟล์แล้วอยากอัปเดต

```bash
git add .
git commit -m "อธิบายว่าแก้อะไร เช่น: update streamlit app"
git push
```

## 4) ถ้า git ถามชื่อ/อีเมล (ครั้งแรกที่ใช้ git บนเครื่อง)

```bash
git config --global user.name "ชื่อของคุณ"
git config --global user.email "อีเมลของคุณ"
```

## 5) ถ้า push แล้วเจอ error ขอ username/password (GitHub ไม่รับ password แล้ว)
ต้องใช้ **Personal Access Token** แทนรหัสผ่าน:
- ไปที่ GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- ติ๊กสิทธิ์ `repo` แล้ว copy token เก็บไว้
- ตอน push ให้ใส่ username ปกติ แต่ช่อง password ให้ **วาง token แทน**

## หมายเหตุ
- ไฟล์ `.gitignore` กันไม่ให้ push `__pycache__/`, `venv/` ขึ้นไปแล้ว
- โฟลเดอร์ `models/*.pkl` มีขนาดไม่ใหญ่มาก (~1.3MB) push ได้ปกติไม่ต้องใช้ Git LFS
- อย่าลืมแก้ข้อมูลใน `app.py` หน้า "ข้อมูลผู้พัฒนา" ให้เป็นข้อมูลจริงก่อน push
