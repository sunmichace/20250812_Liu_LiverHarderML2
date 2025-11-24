#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证PDF导出功能
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 确保输出目录存在
os.makedirs('data/processed', exist_ok=True)

# 创建一个简单的测试图表
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.title('PDF Export Test')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# 保存PNG和PDF格式
plt.savefig('data/processed/test_export.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/test_export.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("✅ PDF导出功能测试成功！")
print("生成文件:")
print("- data/processed/test_export.png")
print("- data/processed/test_export.pdf")