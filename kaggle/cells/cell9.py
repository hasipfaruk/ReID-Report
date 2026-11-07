# ===== CELL 9 : ONE-CLICK DOWNLOAD  zip of all report files (tables, graphs, images, logs) =====
# Creates /kaggle/working/report_files.zip and shows a download link under this cell.
# Model weights (.pth) are left out because they are large; set INCLUDE_WEIGHTS = True to add *_best.pth.
import zipfile
from IPython.display import FileLink, display

INCLUDE_WEIGHTS = False
zip_path = '/kaggle/working/report_files.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(rc.OUT_DIR)):
        if f.endswith(('_ckpt.pth', '.tmp')) or (f.endswith('.pth') and not (INCLUDE_WEIGHTS and f.endswith('_best.pth'))):
            continue
        z.write(os.path.join(rc.OUT_DIR, f), arcname=f'outputs/{f}')
        print('  added', f)
print(f'{zip_path}: {os.path.getsize(zip_path) / 1e6:.1f} MB')
os.chdir('/kaggle/working')
display(FileLink('report_files.zip'))
