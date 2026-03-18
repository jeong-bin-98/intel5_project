import torch
print(torch.cuda.is_available())  # True면 성공!
print(torch.cuda.get_device_name(0))  # GPU 이름 출력
