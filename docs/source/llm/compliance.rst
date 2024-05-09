Compliance
===========

Guardrails
***********

MaxGaurdRails
^^^^^^^^^^^^^^
MaxGaurdRails is a class that applies a set of validations on User Prompt.

Args:
    - ``llm``: The language model to use for validation.
    - ``services_information (str, optional)``: A description of the services provided by the chat bot. Defaults to "Generic Retrieval Augmented chat bot based on input documents".
    - ``pass_if_invalid (bool, optional)``: Whether to pass the validation if the response from the language model is invalid. Defaults to ``True``.

Attributes:
    - ``llm``: The language model to use for validation.
    - ``services_information (str)``: A description of the services provided by the chat bot.
    - ``pass_if_invalid (bool)``: Whether to pass the validation if the response from the language model is invalid.

Methods:
    - ``get_validation_prompt``: Generates the prompt to send to the language model.

        - ``user_message (str)``: The user message to validate.

    - ``get_llm_response_async``: Gets the response from the language model asynchronously.

        - ``prompt (str)``: The prompt to send to the language model.

    - ``validate_asyc``: Validation method for the ResponseEvaluator.

        - ``value (Any)``: The value to validate.

    - ``get_llm_response``: Gets the response from the language model.

        - ``prompt (str)``: The prompt to send to the language model.

    - ``validate``: Validation method for the ResponseEvaluator.

        - ``value (Any)``: The value to validate.

Raises:
    - ``RuntimeError``: If there is an error getting a response from the language model.

Returns:
    - ``get_validation_prompt``: The prompt to send to the language model.
    - ``get_llm_response_async``, ``get_llm_response``: The response from the language model.
    - ``validate_asyc``, ``validate``: True if the validation passes, False otherwise.
    
.. code-block:: python

    from maxaillm.compliance.guardrails import MaxGaurdRails
    from maxaillm.model.llm import MaxAnthropicLLM
    
    # initialize LLM
    llm = MaxAnthropicLLM("claude-2.1")
    query = "Hell World!"
    
    # initialize MaxGuardrails
    guardrails = MaxGaurdRails(llm=llm)
    valid = guardrails.validate(query)
    
    if not valid:
        print("the question asked doesn't abide to usage compliances")
        