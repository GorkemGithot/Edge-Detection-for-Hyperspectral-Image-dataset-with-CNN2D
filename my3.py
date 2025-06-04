import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ======================================
# 1) HiperSpektral Görüntü Oluşturma
# ======================================

folder_path = 'sugar_salt_flour_contamination'  

png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
num_bands = len(png_files)
assert num_bands == 96, f"Bulunan dosya sayısı: {num_bands}. Beklenen: 96."

sample = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
assert sample is not None, f"{png_files[0]} okunamadı!"
H, W = sample.shape

band_list = []
for file in png_files:
    img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    assert img_gray.shape == (H, W), f"{file} boyutu {img_gray.shape} değil!"
    img_f = img_gray.astype(np.float32) / 255.0
    band_list.append(img_f)

img_hsi_np = np.stack(band_list, axis=-1)  # (H, W, 96)
print(f"Hiperspektral veri oluşturuldu. Boyut: {img_hsi_np.shape}")

# ======================================
# 2) Spektral PCA (Boyut İndirgeme)
# ======================================

X = img_hsi_np.reshape(-1, 96)  # (H*W, 96)
n_components = 10
pca = PCA(n_components=n_components, whiten=False)
X_pca = pca.fit_transform(X)  # (H*W, 10)
img_hsi_pca_np = X_pca.reshape(H, W, n_components)  # (H, W, 10)
print(f"PCA uygulandı. Yeni veri boyutu: {img_hsi_pca_np.shape}")

# ======================================
# 3) Torch Tensor’a Çevirme (Channel First)
# ======================================

img_hsi = torch.from_numpy(img_hsi_pca_np).permute(2, 0, 1).contiguous().float()
C = img_hsi.shape[0]
device = torch.device('cpu')  # İstersen 'cuda' yapabilirsin
img_hsi = img_hsi.to(device)

# ======================================
# 4) Laplacian Benzeri Kernel ve Target Hesapla
# ======================================

laplacian_np = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

laplacian = torch.from_numpy(laplacian_np).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    target_channels = []
    for c in range(C):
        channel = img_hsi[c:c+1, :, :].unsqueeze(0)   # (1,1,H,W)
        conv = F.conv2d(channel, laplacian, padding=1)
        conv = conv.squeeze(0).squeeze(0)
        target_channels.append(torch.abs(conv))
    target_hsi = torch.stack(target_channels, dim=0)
    max_all = target_hsi.max()
    if max_all > 0:
        target_hsi = target_hsi / max_all

print(f"Target (Laplacian) hesaplandı. Boyut: {target_hsi.shape}")

# ======================================
# 5) Öğrenilecek Kernel Parametresi
# ======================================

kernel = torch.randn(3, 3, dtype=torch.float32, device=device, requires_grad=True)

# ======================================
# 6) Eğitim Döngüsü
# ======================================

lr = 0.01
num_epochs = 250

for epoch in range(num_epochs):
    w = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
    pred_channels = []
    for c in range(C):
        channel = img_hsi[c:c+1, :, :].unsqueeze(0)
        conv = F.conv2d(channel, w, padding=1)
        conv = conv.squeeze(0).squeeze(0)
        pred_channels.append(conv)
    pred_hsi = torch.stack(pred_channels, dim=0)
    loss = torch.mean((pred_hsi - target_hsi) ** 2)
    loss.backward()
    with torch.no_grad():
        kernel -= lr * kernel.grad
        kernel.grad.zero_()
    print(f"Epoch {epoch+1:3d}/{num_epochs}, Loss: {loss.item():.6f}")

print("Eğitim tamamlandı.")
print("Öğrenilen 3×3 Kernel:")
print(kernel.detach().cpu().numpy())

# ======================================
# 7) Final Prediction ve Normalize
# ======================================

with torch.no_grad():
    w = kernel.unsqueeze(0).unsqueeze(0)
    final_pred_channels = []
    for c in range(C):
        channel = img_hsi[c:c+1, :, :].unsqueeze(0)
        conv = F.conv2d(channel, w, padding=1)
        conv = torch.abs(conv.squeeze(0).squeeze(0))
        final_pred_channels.append(conv)
    final_pred_hsi = torch.stack(final_pred_channels, dim=0)
    max_fp = final_pred_hsi.max()
    if max_fp > 0:
        final_pred_hsi = final_pred_hsi / max_fp

print(f"Final prediction boyutu: {final_pred_hsi.shape}")

# ======================================
# 8) Görselleştirme (Orijinal ve Tahmin)
# ======================================

band_index = 0
orig_band = img_hsi[band_index].cpu().numpy()
pred_band = final_pred_hsi[band_index].cpu().numpy()
tgt_band  = target_hsi[band_index].cpu().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(orig_band, cmap='gray')
plt.title(f'Orijinal (PCA sonrası) - Band {band_index}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_band, cmap='gray')
plt.title(f'Prediction - Band {band_index}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(tgt_band, cmap='gray')
plt.title(f'Target (Laplacian) - Band {band_index}')
plt.axis('off')

plt.tight_layout()
plt.show()


pred_np = pred_band  # Zaten normalize edilmiş [0,1]


edge_uint8 = (pred_np * 255).astype(np.uint8)


_, binary_edges = cv2.threshold(edge_uint8, 40, 255, cv2.THRESH_BINARY)


contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


contour_image = cv2.cvtColor(edge_uint8, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  


contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(pred_np, cmap='gray')
plt.title('Prediction (kenar haritası)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_edges, cmap='gray')
plt.title('Threshold ile Kenarlar')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(contour_image_rgb)
plt.title('Konturlar ile Çizilmiş')
plt.axis('off')

plt.tight_layout()
plt.show()