import os
import glob

def cleanup():
    """
    清理临时仿真文件（.inp, .rpt, .bin）。
    """
    patterns = ['sim_*.inp', 'sim_*.rpt', 'sim_*.bin']
    count = 0
    for pat in patterns:
        files = glob.glob(pat)
        for f in files:
            try:
                os.remove(f)
                count += 1
            except Exception as e:
                print(f"无法移除文件 {f}: {e}")
    print(f"清理了 {count} 个临时文件。")

if __name__ == '__main__':
    cleanup()
