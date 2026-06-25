# gen_code.py
from app import app, db, InvitationCode
import random
import string


def generate_more_codes(count=5):
    with app.app_context():
        print(f"正在生成 {count} 个新邀请码...")
        new_list = []
        for _ in range(count):
            code_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            db.session.add(InvitationCode(code=code_str))
            new_list.append(code_str)

        db.session.commit()
        print("✅ 生成成功！请查收：")
        for c in new_list:
            print(f"🔥 {c}")


if __name__ == '__main__':
    # 可以在括号里改数字，想生几个生几个
    generate_more_codes(5)