import re
import sys
import os
import logging
from datetime import datetime
import requests
from AppConfig import build_github_headers
import json
import zipfile
import tempfile
import shutil
import markdown
from pathlib import Path
from packaging import version
from diagnostics import report_error
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QLabel, QProgressBar, QTextEdit,
                            QMessageBox, QGroupBox, QDialog, QDialogButtonBox,
                            QMenu, QMenuBar, QAction, QTabWidget, QTextBrowser)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView  # 用于渲染Markdown


class UpdateDialog(QDialog):
    """更新对话框"""
    download_progress = pyqtSignal(int, int)
    update_status = pyqtSignal(str,str)
    def __init__(self, parent=None, startup_check=False):
        super().__init__(parent)
        self.parent = parent
        self.startup_check = startup_check  # 是否为启动时检查
        self.update_checker = None
        self.update_downloader = None
        self.update_info = None
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("检查更新")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # 更新状态选项卡
        self.status_tab = QWidget()
        status_layout = QVBoxLayout(self.status_tab)

        # 日志区域
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        status_layout.addWidget(log_group)

        # 状态信息
        self.status_label = QLabel("点击检查更新按钮开始")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("关于本软件及更新日志："))
        url_label = QLabel("""
                        <style>
                            a { color: blue; text-decoration: none; }
                            a:hover { color: red; text-decoration: underline; }
                        </style>
                        <a href="https://github.com/CSSAcslin/LifeCalor/releases">请点击访问 Github 本项目页面</a>
                        """)
        url_label.setOpenExternalLinks(True)
        url_layout.addWidget(url_label)
        status_layout.addLayout(url_layout)
        status_layout.addStretch()

        self.tab_widget.addTab(self.status_tab, "更新状态")

        # 更新说明选项卡（初始为空，有更新时填充）
        self.release_notes_tab = QWidget()
        notes_layout = QVBoxLayout(self.release_notes_tab)

        self.release_notes_browser = QTextBrowser()
        self.release_notes_browser.setOpenExternalLinks(True)  # 允许打开外部链接
        notes_layout.addWidget(self.release_notes_browser)

        self.tab_widget.addTab(self.release_notes_tab, "更新说明")

        # 按钮区域
        button_layout = QHBoxLayout()

        self.check_button = QPushButton("检查更新")
        self.check_button.clicked.connect(self.check_for_updates)
        button_layout.addWidget(self.check_button)

        self.update_button = QPushButton("立即更新")
        self.update_button.clicked.connect(self.start_update)
        self.update_button.setEnabled(False)
        button_layout.addWidget(self.update_button)

        if self.startup_check:
            self.remind_button = QPushButton("稍后提醒")
            self.remind_button.clicked.connect(self.remind_later)
            button_layout.addWidget(self.remind_button)
            self.never_button = QPushButton("永不提醒")
            self.never_button.setStyleSheet("""QPushButton {color: rgba(0, 0, 0, 172);}""")
            self.never_button.clicked.connect(self.remind_never)
            button_layout.addWidget(self.never_button)

        else:
            self.close_button = QPushButton("关闭")
            self.close_button.clicked.connect(self.close)
            button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)


    def log_message(self, message, error = False):
        """添加日志消息(与外部串联，同时有一个label在dialog里)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        if not error:
            logging.info(f"程序更新操作：{message}")
        else:
            logging.error(message)

    def check_for_updates(self):
        """检查更新"""
        self.check_button.setEnabled(False)
        self.update_button.setEnabled(False)
        self.status_label.setText("正在检查更新...")
        self.log_message("开始检查更新")

        # 创建并启动检查线程
        self.update_checker = UpdateChecker(
            self.parent.repo_owner,
            self.parent.repo_name,
            self.parent.current_version,
            self.parent.PAT,
        )
        self.update_checker.update_status.connect(self.update_status)
        self.update_checker.error_occurred.connect(self.handle_check_error)
        self.update_checker.version_info.connect(self.handle_version_info)
        self.update_checker.start()

    def handle_check_error(self, error_message):
        """处理检查错误"""
        self.check_button.setEnabled(True)
        self.status_label.setText("检查更新失败")
        self.update_status.emit("检查更新失败",'error')
        self.log_message(f"更新错误: {error_message}",error=True)

        # 显示错误对话框
        report_error(self, "检查更新失败", error_message)

    def handle_version_info(self, version_info):
        """处理版本信息"""
        self.check_button.setEnabled(True)

        if version_info.get('update_available'):
            self.update_info = version_info
            latest_version = version_info['latest_version']
            self.status_label.setText(f"发现新版本: v{latest_version}")
            self.log_message(f"发现新版本! v{latest_version}")
            self.update_status.emit("发现新版本可更新！",'idle')

            # 显示更新说明（支持Markdown）
            self.show_release_notes(version_info.get('release_notes', '暂无更新说明'))

            # 启用更新按钮
            self.update_button.setEnabled(True)

            # 自动切换到更新说明选项卡
            self.tab_widget.setCurrentIndex(1)

            # 如果是启动时检查，自动弹窗提示
            if self.startup_check:
                self.show()
                self.raise_()
                self.activateWindow()

        else:
            self.status_label.setText("当前已是最新版本")
            self.log_message("当前已是最新版本")
            QMessageBox.information(self, "更新检查", "您的程序已是最新版本")

    def show_release_notes(self, notes):
        """显示更新说明，支持Markdown格式"""
        try:
            # 尝试将Markdown转换为HTML
            html_content = markdown.markdown(notes)
            self.release_notes_browser.setHtml(html_content)
        except Exception as e:
            # 如果转换失败，直接显示纯文本
            self.log_message(f"Markdown解析失败，使用纯文本: {str(e)}",error=True)
            self.release_notes_browser.setPlainText(notes)

    def start_update(self):
        """开始下载更新"""
        if not self.update_info:
            return

        reply = QMessageBox.question(
            self,
            "确认更新",
            f"发现新版本 v{self.update_info['latest_version']}，是否立即下载并安装？\n\n"
            "注意：更新过程中程序将自动重启。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.check_button.setEnabled(False)
        self.update_button.setEnabled(False)
        if self.startup_check:
            self.remind_button.setEnabled(False)

        self.log_message(f"开始下载更新文件: {self.update_info['file_name']}")
        self.update_status.emit("开始下载更新文件", 'working')

        # 创建并启动下载线程
        self.update_downloader = UpdateDownloader(
            self.update_info['download_url'],
            self.update_info['file_name'],
            self.parent.PAT,
        )
        self.update_downloader.download_progress.connect(self.download_progress)
        self.update_downloader.download_status.connect(self.update_status)
        self.update_downloader.download_complete.connect(self.handle_download_complete)
        self.update_downloader.download_error.connect(self.handle_download_error)
        self.update_downloader.start()

    def remind_later(self):
        """稍后提醒"""
        logging.info("更新会在 1 天后提醒")
        self.reject()

    def remind_never(self):
        """永不更新"""
        reply = QMessageBox.question(
            self,
            "确认操作",
            f"是否永远都不检查更新"
            "注意：此操作不可撤销，无法自动获取最新更新，\n以后只可以选择手动更新",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            self.parent.settings.setValue("should_check", False)
            self.close()
            return

    def handle_download_error(self, error_message):
        """处理下载错误"""
        self.check_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self.status_label.setText("下载失败")
        self.log_message(f"下载错误: {error_message}",error=True)

        report_error(self, "下载失败", error_message)

    def handle_download_complete(self, file_path):
        """处理下载完成"""
        self.status_label.setText("下载完成，准备应用更新")
        self.log_message("更新文件下载完成")

        # 询问用户是否立即应用更新
        reply = QMessageBox.question(
            self,
            "下载完成",
            "更新文件下载完成，是否立即应用更新？\n程序将在更新后自动重启（旧版程序需要手动删除）。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.apply_update(file_path)
        else:
            self.check_button.setEnabled(True)
            self.update_button.setEnabled(True)

    def apply_update(self, update_file_path):
        """应用更新"""
        try:
            self.status_label.setText("正在应用更新...")
            self.log_message("开始应用更新")

            # 获取应用程序目录和路径
            app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
            app_path = sys.argv[0]
            app_name = os.path.basename(app_path)
            new_path = re.sub(r'-v\d+\.\d+\.\d+', f'-v{self.update_info['latest_version']}', app_path)
            # 创建临时目录用于解压
            temp_dir = tempfile.mkdtemp()
            self.log_message(f"创建临时目录: {temp_dir}")

            # 应用更新
            success, message = UpdateApplier.apply_update(update_file_path, temp_dir)

            if success:
                self.log_message("更新文件解压成功，创建批处理脚本")

                # 创建批处理脚本来替换文件并重启
                bat_content = f"""
@echo off
chcp 65001 > nul
echo 正在准备更新...
timeout /t 2 /nobreak > nul

echo 正在关闭应用程序...
taskkill /f /im "{app_name}" > nul 2>&1

echo 正在应用更新...
xcopy /Y /E /I "{temp_dir}\\*" "{app_directory}\\"

echo 正在清理临时文件...
rmdir /s /q "{temp_dir}"

echo 启动新版本...
cd /d "{app_directory}"
start "" "{new_path}"

echo 更新完成!
del "%~f0"
"""

                # 保存批处理文件到临时目录
                bat_path = os.path.join(tempfile.gettempdir(), "update_app.bat")
                with open(bat_path, 'w', encoding='utf-8') as f:
                    f.write(bat_content)

                self.log_message("启动更新脚本")

                # 启动批处理脚本
                import subprocess
                CREATE_NO_WINDOW = 0x08000000  # 不显示命令行窗口
                subprocess.Popen([bat_path], creationflags=CREATE_NO_WINDOW)

                # 退出当前程序
                self.log_message("程序即将重启...")
                QTimer.singleShot(1000, QApplication.quit)

            else:
                self.handle_update_error(message)
                # 清理临时目录
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            self.handle_update_error(f"应用更新时发生错误: {str(e)}")

    def handle_update_error(self, error_message):
        """处理更新错误"""
        self.check_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self.status_label.setText("更新失败")
        self.update_status.emit("更新失败",'error')
        self.log_message(f"更新错误: {error_message}",error = True )

        report_error(self, "更新失败", error_message)


class UpdateChecker(QThread):
    """后台线程：检查GitHub是否有更新"""

    # 定义信号：更新状态、错误信息、版本信息
    update_status = pyqtSignal(str,str)
    error_occurred = pyqtSignal(str)
    version_info = pyqtSignal(dict)
    check_completed = pyqtSignal(bool)  # 检查完成时发射，参数表示是否有更新

    def __init__(self, repo_owner, repo_name, current_version, PAT):
        super().__init__()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.current_version = current_version
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
        self.headers = build_github_headers({
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28'})

    def run(self):
        """主线程执行函数"""
        try:
            self.update_status.emit("正在连接GitHub服务器...",'working')

            # 发送API请求
            response = requests.get(self.api_url, headers=self.headers, timeout=30)

            # 检查响应状态
            if response.status_code == 403:
                # GitHub API 限制
                remaining_requests = response.headers.get('X-RateLimit-Remaining', '未知')
                self.error_occurred.emit(f"API请求受限，剩余次数: {remaining_requests}")
                self.check_completed.emit(False)
                return
            elif response.status_code != 200:
                self.error_occurred.emit(f"服务器错误: {response.status_code}")
                return

            data = response.json()[0]

            # 解析版本信息
            latest_version = data.get('tag_name', '').lstrip('v')
            if not latest_version:
                self.error_occurred.emit("无法解析版本信息")
                self.check_completed.emit(False)
                return

            # 查找可下载资源
            download_info = self.find_download_asset(data.get('assets', []))
            if not download_info:
                self.error_occurred.emit("未找到可用的更新文件")
                self.check_completed.emit(False)
                return

            # 比较版本
            if version.parse(latest_version) > version.parse(self.current_version):
                update_info = {
                    'update_available': True,
                    'latest_version': latest_version,
                    'download_url': download_info['url'],
                    'file_name': download_info['name'],
                    'release_notes': data.get('body', '暂无更新说明')
                }
                self.check_completed.emit(True)
            else:
                update_info = {'update_available': False}
                self.check_completed.emit(False)

            self.version_info.emit(update_info)

        except requests.exceptions.Timeout:
            self.error_occurred.emit("连接超时，请检查网络连接")
            self.check_completed.emit(False)
        except requests.exceptions.ConnectionError:
            self.error_occurred.emit("网络连接错误")
            self.check_completed.emit(False)
        except Exception as e:
            self.error_occurred.emit(f"检查更新时发生未知错误: {str(e)}")
            self.check_completed.emit(False)

    def find_download_asset(self, assets):
        """在assets中查找合适的下载文件"""
        # 优先查找zip文件，然后是exe文件
        for asset in assets:
            name = asset.get('name', '').lower()
            if name.endswith('.zip') or name.endswith('.exe'):
                return {
                    'name': asset.get('name'),
                    'url': asset.get('browser_download_url')
                }
        return None


class UpdateDownloader(QThread):
    """后台线程：下载更新文件"""

    # 定义信号：下载进度、状态、完成、错误
    download_progress = pyqtSignal(int,int)
    download_status = pyqtSignal(str,str)
    download_complete = pyqtSignal(str)  # 文件路径
    download_error = pyqtSignal(str)

    def __init__(self, download_url, file_name,PAT):
        super().__init__()
        self.download_url = download_url
        self.file_name = file_name
        self.temp_dir = tempfile.gettempdir()
        self.headers = build_github_headers()

    def run(self):
        """下载主线程"""
        try:
            self.download_status.emit("开始下载更新文件...",'working')

            # 准备请求头


            # 流式下载，支持进度显示
            response = requests.get(self.download_url, headers=self.headers, stream=True, timeout=60)
            response.raise_for_status()

            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            # 保存文件路径
            file_path = os.path.join(self.temp_dir, self.file_name)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)

                        # 计算并发送进度
                        if total_size > 0:
                            self.download_progress.emit(1, total_size)
                            self.download_progress.emit(downloaded_size,total_size)

            self.download_status.emit("下载完成",'idle')
            self.download_complete.emit(file_path)
            self.download_progress.emit(total_size+1,total_size)

        except requests.exceptions.Timeout:
            self.download_error.emit("下载超时")
        except requests.exceptions.ConnectionError:
            self.download_error.emit("网络连接中断")
        except Exception as e:
            self.download_error.emit(f"下载失败: {str(e)}")


class UpdateApplier:
    """应用更新工具类"""

    @staticmethod
    def apply_update(update_file_path, app_directory):
        """应用更新"""
        try:
            # 检查文件类型并相应处理
            if update_file_path.lower().endswith('.zip'):
                return UpdateApplier.apply_zip_update(update_file_path, app_directory)
            else:
                return UpdateApplier.apply_file_update(update_file_path, app_directory)

        except Exception as e:
            return False, f"应用更新失败: {str(e)}"

    @staticmethod
    def apply_zip_update(zip_path, app_directory):
        """处理ZIP格式更新"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(app_directory)
            return True, "更新应用成功"
        except zipfile.BadZipFile:
            return False, "更新文件损坏"
        except Exception as e:
            return False, f"解压失败: {str(e)}"

    @staticmethod
    def apply_file_update(file_path, app_directory):
        """处理单个文件更新"""
        try:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(app_directory, file_name)
            shutil.copy2(file_path, dest_path)
            return True, "文件更新成功"
        except Exception as e:
            return False, f"文件复制失败: {str(e)}"
