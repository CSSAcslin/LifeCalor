from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisAction:
    action_id: str
    category: str
    title: str
    handler: str
    page: str = ""
    input_kind: str = "video"
    processed_types: tuple[str, ...] = ()
    roi: bool = False


ANALYSIS_ACTIONS = (
    AnalysisAction("preprocess.em", "分析预处理", "EM 背景归一化…", "pre_process_EM", "em"),
    AnalysisAction("statistics.roi", "统计与信号提取", "ROI 时序统计…", "process_roi_statistics", roi=True),
    AnalysisAction("statistics.mean", "统计与信号提取", "全场平均时序…", "process_signal_avg"),
    AnalysisAction("statistics.distribution", "统计与信号提取", "当前帧 ROI 值分布…", "process_roi_value_distribution", input_kind="image_or_video", roi=True),
    AnalysisAction("statistics.time_mean", "统计与信号提取", "时间平均振幅图…", "process_atam"),
    AnalysisAction("transform.stft_quality", "时频与频域分析", "STFT 质量分析…", "quality_EM_stft", processed_types=("EM_pre_processed",)),
    AnalysisAction("transform.stft", "时频与频域分析", "STFT 时频计算…", "process_EM_stft", processed_types=("EM_pre_processed",)),
    AnalysisAction("transform.cwt_quality", "时频与频域分析", "CWT 质量分析…", "quality_EM_cwt", processed_types=("EM_pre_processed",)),
    AnalysisAction("transform.cwt", "时频与频域分析", "CWT 时频计算…", "process_EM_cwt", processed_types=("EM_pre_processed",)),
    AnalysisAction("transform.fft2", "时频与频域分析", "二维傅里叶变换…", "process_tDFT", input_kind="image_or_video"),
    AnalysisAction("transform.ifft2", "时频与频域分析", "二维傅里叶逆变换…", "process_tDiFT", input_kind="spectrum"),
    AnalysisAction("lifetime.region", "寿命与物理参数", "区域指数寿命拟合…", "region_analyze_start", "region"),
    AnalysisAction("lifetime.map", "寿命与物理参数", "像素级寿命热图…", "distribution_analyze_start", "lifetime"),
    AnalysisAction("physics.diffusion", "寿命与物理参数", "扩散系数分析…", "vectorROI_signal_show", "diffusion"),
    AnalysisAction("physics.heat", "寿命与物理参数", "传热系数热图…", "heat_transfer_start", "heat", input_kind="video_or_lifetime"),
    AnalysisAction("channel.gaussian", "单通道电生理", "二维高斯拟合信号提取…", "process_tDgf", processed_types=("Roi_applied",)),
    AnalysisAction("channel.simple", "单通道电生理", "简单单通道提取…", "process_simple_scs", processed_types=("Roi_applied",)),
    AnalysisAction("motion.heartbeat", "", "心肌细胞运动分析…", "process_heartbeat"),
)

ACTION_BY_ID = {action.action_id: action for action in ANALYSIS_ACTIONS}
