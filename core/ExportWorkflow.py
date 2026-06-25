def save_dataframe(dataframe, file_path: str, has_header: bool, task_state) -> bool:
    task_state.start()
    try:
        if file_path.lower().endswith(".csv"):
            dataframe.to_csv(file_path, index=False, header=has_header)
        else:
            dataframe.to_csv(file_path, sep="\t", index=False, header=has_header)
        task_state.complete()
        return True
    except Exception as exc:
        task_state.fail(str(exc))
        return False
