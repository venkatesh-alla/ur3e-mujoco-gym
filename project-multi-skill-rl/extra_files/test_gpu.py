import torch

if torch.cuda.is_available():
	print("GPU skills acquired!!!")
	print(torch.cuda.get_device_name(0))

else:
	print('NO GPU Available')


# Setting up Tf32
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True	


print(f"PyTorch's cuda version: {torch.version.cuda}")
print(f"NCCL version: {torch.cuda.nccl.version()}")

print("Torch cudnn availability:")
print(torch.backends.cudnn.is_available())  # Should return True
print(torch.backends.cudnn.version())  # Should return an integer (e.g., 8401)

print("Get device capability:")
print(torch.cuda.get_device_capability()) # Should return (8,6)

print("Allows Tf32:")
print(torch.backends.cuda.matmul.allow_tf32)  # Should print False
print(torch.backends.cudnn.allow_tf32)         # Should print True


print("A5000 specs:")
print(torch.cuda.get_device_properties(0))
