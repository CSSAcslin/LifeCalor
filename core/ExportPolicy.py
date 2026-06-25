from AppConfig import is_em_frequency_result


def can_export_em_data(processed_type) -> bool:
    return is_em_frequency_result(processed_type)


def prepare_dataframe_for_export(dataframe, current_mode: str, include_fitting: bool):
    if include_fitting:
        return dataframe
    if current_mode == "curve":
        return dataframe.loc[:, dataframe.columns != "fit_curve"]
    if current_mode == "diff":
        return dataframe.loc[:, dataframe.columns.get_level_values(1) != "拟合曲线"]
    return dataframe
