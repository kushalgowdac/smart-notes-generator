import sys
print("Python executable:", sys.executable)
print("\nPython path:")
for p in sys.path:
    print(" ", p)

print("\n" + "="*60)
print("Testing paddle import...")
try:
    import paddle
    print(f"✓ Paddle imported successfully! Version: {paddle.__version__}")
except Exception as e:
    print(f"✗ Paddle import failed: {e}")

print("\n" + "="*60)
print("Testing paddleocr import...")
try:
    from paddleocr import PaddleOCR
    print("✓ PaddleOCR imported successfully!")
except Exception as e:
    print(f"✗ PaddleOCR import failed: {e}")
    import traceback
    traceback.print_exc()
