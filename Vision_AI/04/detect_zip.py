import cv2
import matplotlib.pyplot as plt

# ���� �̹������� ���� ��ȣ�� �����ϴ� �Լ�
def detect_zipno(fname):
    # �̹��� �о� ���̱�
    img = cv2.imread(fname)
    # �̹��� ũ�� ���ϱ�
    h, w = img.shape[:2]
    # �̹����� ������ ���κи� �����ϱ� --- (*1)
    img = img[0:h//2, w//3:]
    
    # �̹��� ����ȭ�ϱ� --- (*2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) 
    im2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
    
    # ���� �����ϱ� --- (*3)
    cnts = cv2.findContours(im2, 
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[1]
    
    # ������ �̹������� ���� �����ϱ�--- (*4)
    result = []
    for pt in cnts:
        x, y, w, h = cv2.boundingRect(pt)
        # �ʹ� ũ�ų� �ʹ� ���� �κ� �����ϱ� --- (*5)
        if not(50 < w < 70): continue
        result.append([x, y, w, h])
    # ������ ������ ��ġ�� ���� �����ϱ� --- (*6)
    result = sorted(result, key=lambda x: x[0])
    # ������ ������ �ʹ� ����� �͵� �����ϱ� --- (*7)
    result2 = []
    lastx = -100
    for x, y, w, h in result:
        if (x - lastx) < 10: continue
        result2.append([x, y, w, h])
        lastx = x
    # �ʷϻ� �׵θ� ����ϱ� --- (*8)
    for x, y, w, h in result2:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    return result2, img

if __name__ == '__main__':
    # �̹����� �����ؼ� �����ȣ �����ϱ�
    cnts, img = detect_zipno("img/hagaki1.png")

    # ��� ����ϱ� --- (*7)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig("detect-zip.png", dpi=200)
    plt.show()