import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import base64
import io
import numpy as np
from PIL import Image
from torchvision import transforms

# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# CIFAR-10 classes
# =====================================================
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# =====================================================
# Model definition
# =====================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet34CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# =====================================================
# Load models (singletons)
# =====================================================
def load_model(path):
    model = ResNet34CIFAR().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

MODELS = {
    "resnet34": load_model("ml_models/resnet_models/trained_models/resnet34_cifar10.pth"),
    "adversarial_resnet": load_model("ml_models/resnet_models/trained_models/resnet34_adv_cifar10.pth"),
    "sam_resnet": load_model("ml_models/resnet_models/trained_models/resnet34_sam_cifar10.pth"),
}

# =====================================================
# Preprocessing
# =====================================================
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def normalize(x):
    return (x - mean) / std

# =====================================================
# Attacks
# =====================================================
def gaussian_noise(x, sigma):
    noise = sigma * torch.randn_like(x)
    return torch.clamp(x + noise, 0, 1), noise

def fgsm_attack(model, x, epsilon):
    x = x.clone().detach().requires_grad_(True)
    logits = model(normalize(x))
    pred = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, pred)
    loss.backward()
    noise = epsilon * x.grad.sign()
    return torch.clamp(x + noise, 0, 1).detach(), noise

def perturb_weights(model, sigma):
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(sigma * torch.randn_like(p))
    return perturbed

# =====================================================
# Utils
# =====================================================
def decode_base64_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def tensor_to_base64(x):
    img = (x.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# =====================================================
# Inference API
# =====================================================
def infer_resnet(model_name, image_b64, attack_type, noise_value):
    base_model = MODELS.get(model_name, MODELS["resnet34"])
    working_model = base_model

    x = decode_base64_image(image_b64)

    if attack_type == "gaussian":
        x_noisy, _ = gaussian_noise(x, noise_value)

    elif attack_type == "fgsm":
        x_noisy, _ = fgsm_attack(base_model, x, noise_value)

    elif attack_type == "weight":
        working_model = perturb_weights(base_model, noise_value)
        x_noisy = x.clone()

    else:
        x_noisy = x.clone()

    with torch.no_grad():
        logits = working_model(normalize(x_noisy))
        probs = F.softmax(logits, dim=1)
        confidence, pred = probs.max(1)

    return {
        "predicted_class": CIFAR10_CLASSES[pred.item()],
        "confidence": float(confidence.item()),
        "noisy_image": f"data:image/png;base64,{tensor_to_base64(x_noisy)}",
        "output_image": f"data:image/png;base64,{tensor_to_base64(x_noisy)}",
    }
