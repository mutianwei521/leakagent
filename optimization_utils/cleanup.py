import os
import glob

def cleanup():
    """
    Clean up temporary simulation files (.inp, .rpt, .bin).
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
                print(f"Cannot remove file {f}: {e}")
    print(f"Cleaned up {count} temporary files.")

if __name__ == '__main__':
    cleanup()
