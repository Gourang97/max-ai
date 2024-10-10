Compliance
===========

Guardrails
***********

MaxGuardRails
^^^^^^^^^^^^^^
Provides various guardrails for validating and safeguarding text inputs.

Args:
    - ``llm (Any, Optional)``: The language model to be used for legacy guardrail support. Defaults to None.

Attributes:
    - ``guard``: An instance of the Guard class for managing validations.
    - ``llm``: The language model for legacy guardrail support.

Methods:
    - ``prompt_injection``: Prevents prompt injection using Rebuff.

    - ``pii``: Adds PII safeguard.

        - Args:
            - ``email_address (bool, Optional)``: Whether to detect email addresses. Defaults to True.
            - ``phone_number (bool, Optional)``: Whether to detect phone numbers. Defaults to True.

    - ``off_topic``: Checks if a text is related to a topic.

        - Args:
            - ``valid_topics (list)``: List of valid topics.
            - ``invalid_topics (list)``: List of invalid topics.
            - ``disable_classifier (bool, Optional)``: Whether to disable the classifier. Defaults to True.
            - ``disable_llm (bool, Optional)``: Whether to disable the language model. Defaults to False.
            - ``kwargs``: Additional keyword arguments.

    - ``detect_gibberish``: Detects non-sensical prompts.

    - ``profanities``: Ensures that there’s no profanity in the text.

    - ``secrets_present``: Detects secrets present in the text.

    - ``validate``: Runs text against validators.

        - Args:
            - ``text (str)``: The text to be validated.

        - Returns:
            - ``tuple[bool, dict]``: A tuple containing the validation status and a dictionary of errors if any.

    - ``_parse_failed_validations``: Extracts information from failed validations.

        - Returns:
            - ``dict``: A dictionary containing the guardrail name and proposed fixed value for each failed validation.

    - ``maxai_guardrails``: Registers MAX.AI validators.

        - Args:
            - ``llm``: The language model.
            - ``validators (list, Optional)``: List of validators to be used. Defaults to DEFAULT_MAXAI_VALIDATORS.
            - ``on_fail (str, Optional)``: Action to take on failure. Defaults to "fix".
            - ``kwargs``: Additional keyword arguments.

    - ``user_defined_guardrail``: Registers a user-defined guardrail.

        - Args:
            - ``custom_guardrail_config (dict)``: Configuration for the custom guardrail.
            - ``llm``: The language model.
            - ``on_fail (str, Optional)``: Action to take on failure. Defaults to "fix".
            - ``kwargs``: Additional keyword arguments.
    
.. code-block:: python

    from maxaillm.compliance.guardrails import MaxGuardRails
    from maxaillm.model.llm import MaxAnthropicLLM
    
    # initialize LLM
    llm = MaxAnthropicLLM("claude-2.1")
    query = "Hell World!"
    
    # initialize MaxGuardrails
    guardrails = MaxGuardRails(llm=llm)
    valid = guardrails.validate(query)
    
    if not valid:
        print("the question asked doesn't abide to usage compliances")
        