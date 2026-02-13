Missing Submit Button

This form has no submit button, which means that user interactions will never be sent to your Streamlit app.

To create a submit button, use the st.form_submit_button() function.

For more information, refer to the documentation for forms.

streamlit.errors.StreamlitMixedNumericTypesError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/churn-workspace/app.py", line 99, in <module>
    voice_minutes = st.number_input("Minutes Vocales", min_value=0.0, value=300)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 532, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/widgets/number_input.py", line 401, in number_input
    return self._number_input(
           ~~~~~~~~~~~~~~~~~~^
        label=label,
        ^^^^^^^^^^^^
    ...<15 lines>...
        ctx=ctx,
        ^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/widgets/number_input.py", line 482, in _number_input
    raise StreamlitMixedNumericTypesError(
        value=value, min_value=min_value, max_value=max_value, step=step
