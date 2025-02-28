# Adding a new dataset to the codebase
After converting a dataset in the Parquet format, you should write a new class extending `CustomDataset` (or any of its children, e.g. `CustomAudioDataset`) in order to use it for training. Additionally, if you want to use such dataset for a previously not supported task, you should also implement a suitable task preparer by extending the `TaskPreparer` class.

Note that `CustomDataset` and `TaskPreparer` should ideally be completely independent from each other. For example don't make any assumption on which dataset you're dealing with inside a `TaskPreparer`, but treat it like a generic dataset. Any dataset-specific parameters (e.g. which dataset fields to rename) can be passed to the `TaskPreparer`'s contructor.

## Extending `CustomDataset`
When subclassing the `CustomDataset` class, you should (re)define the following class attributes:
```python
name: str  # official name of the dataset, can contain uppercase characters too
codename: str  # codename of the dataset, all lowercase
splits: List[str] = ["train", "eval", "test"]
text_keys: List[str] = ["text"]  # names of text columns in the dataset
optional_filters: List[str] = []  # zero or more of [CustomDataset.CTC_ALIGNER_FILTER, CustomDataset.NUM_WORDS_FILTER]
columns_to_cast: Dict[str, Any] = dict()  # columns that should be casted to specific types
```

For instance, here's how we could write a wrapper class for a dataset called "Acme":
```python
name = "Acme"
codename = "acme"
optional_filters = [CustomDataset.CTC_ALIGNER_FILTER]
columns_to_cast = {"speaker_id": Value(dtype="string", id=None)}
```

Note that we did not redefine some of the superclass's attributes such as `splits`, meaning that we'll inherit it from the base class.

In addition to (re)defining class attributes, you should update the `preparers` field in the class constructor depending on which tasks your dataset is meant to support.

**NOTE**: it is important that you do this before calling the superclass constructor.
For example:
```python
def __init__(...):
    self.preparers = {"ASR": SpeechRecognitionPreparer()}
    super().__init__(...)
```
(you'll see how to write a proper `TaskPreparer` class later in [Extending `TaskPreparer`](#extending-taskpreparer))

Most of the time, i.e. when you're adding a dataset that supports only existing tasks, all you have to do is to implement the `_get_dataset_path` method, in which you have to return the path to the actual Parquet file backing that specific `CustomDataset` instance given the root directory of the dataset as well as other parameters. For example, this is a possible implementation:
```python
def _get_dataset_path(
    self,
    dataset_dir: str,
    source_language: str,
    target_language: str,
    partition_name: str,
    partition_spec: dict,
) -> str:
    filename = f"covost2.{source_language}_{target_language}.{partition_name}.parquet"
    dataset_path = Path(dataset_dir, filename)
    if not dataset_path.exists():
        subdir = "en_xx" if source_language == "en" else "xx_en"
        dataset_path = Path(dataset_dir, subdir, filename)

    return str(dataset_path)
```
if needed, you can also override other methods in the `CustomDataset` class.


### Extending `CustomDataset`'s subclasses
If you find that extending one of `CustomDataset` subclasses is more appropriate (e.g. because yours is an audio dataset), keep in mind that they usually introduce new class attributes that you should (re)define.

For instance, the `CustomTemporalDataset` is the base class for sequence datasets, such as audio and video datasets with associated transcription. This class introduces these additional class attributes:
```python
sequence_keys: List[str] = ["sequence"]
duration_keys: List[str] = ["duration"]
token_rate_validation_triplets = [("duration", "text", 700)]  # (duration_key, text_keys, max_tokens_per_minute)
```

`token_rate_validation_triplets` is used to filter out examples where the sequence-transcription alignment is messed up. For example, `("duration", "text", 700)` means that if the `"text"` field contains more than 700 tokens per minute (rescaled by the value of the `"duration"` field), the example is discarded.

`CustomTemporalDataset` also requires you to update the `duration_getters` attribute in the class constructor. This will be used to understand which function to use to compute the duration of a given sequence type. For example, the `CustomAudioDataset` class (which is a specialization of `CustomTemporalDataset` for audios) defines it as follows:
```python
def __init__(...):
    self.duration_getters.update({"audio": get_audio_duration})
    super.__init__()
```
Similarly, the `CustomVideoDataset` class does:
```python
self.duration_getters.update({"video": get_video_duration})
```

## Extending `TaskPreparer`
Task preparers are the objects that are responsible for preparing the samples in a dataset for a given task. A task preparer defines a set of fields (columns) that should be present in the examples _after_ the preparation has finished. The base task preparer defines the following fields:
```python
final_example_fields = [
    "conversations",
    "source_language",
    "target_language",
    "task",
    "few_shot_examples",
]
```
- `"conversations"` represents the ground truth back-and-forth conversation with the LLM. For example, if the task is ASR, the conversation would start with the user asking something like "please provide a transcription for this audio" and continue with the LLM prividing the correct speech transcription
- `"source_language"` and `"target_language"` are the source and target languages of the task. For example, if the task is ST, `"source_language"` could be "en" and `"target_language"` could be "fr"
- `"task"` is the name of the task that the LLM is being asked to perform
- `"few_shot_examples"` (optional) is a list of few-shot examples and associated ground truths that are used to help the LLM understand the task

It is often the case that your dataset has all the required fields for a given task, just with a different name. For example, the `AudioInputTaskPreparer` requires the dataset to have the "transcription" field, but your dataset might have a "transcript" field instead. In this case, you can use the `remap_keys` argument in the `TaskPreparer`'s constructor to remap the field names. So in the <u>dataset's</u> constructor, you would do:
```python
self.preparers = {"ASR": AudioInputTaskPreparer(remap_keys={"transcript": "transcription"})}
```

Most of the time, all you have to do to implement a task preparer for a new task is to subclass `TaskPreparer` and implement the `_get_input_and_output` method. This method should return the input (user prompt) and output (LLM response) of the task for a given example. For example, this is how it's implemented for the `VisualSpeechRecognitionPreparer` class:
```python
def _get_input_and_output(
    self,
    example,
    source_language: str,
    target_language: str,
    config: CustomDatasetConfig,
):
    prompt_language = random.choice([source_language, "en"])
    input_, desired_output_prefix = "", ""
    if config.INPUTS_TEXT_LIST is not None:
        input_ = random.choice(
            config.INPUTS_TEXT_LIST[config.task][prompt_language]
        )
    if config.OUTPUTS_TEXT_LIST is not None:
        desired_output_prefix = random.choice(
            config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
        )

    return (
        DEFAULT_VIDEO_INPUT_START_TOKEN
        + DEFAULT_VIDEO_INPUT_END_TOKEN
        + "\n"
        + input_,
        desired_output_prefix + example["video_transcription"],
    )
```
As you can see above, usually the input prompt and output response are built by accessing `INPUTS_TEXT_LIST` and `OUTPUTS_TEXT_LIST` from the `config` argument. For each task, `INPUTS_TEXT_LIST` and `OUTPUTS_TEXT_LIST` contain some text templates for the user prompt and the (prefix of the) LLM response, respectively. For example, for the ASR task, we might have "Please provide a transcription for this audio" and "Please transcribe the following audio" among the values in `INPUTS_TEXT_LIST`, and "The transcription is: " and "Here's the transcription: " among the values in `OUTPUTS_TEXT_LIST`.

**NOTE #1**: if you're creating a new task, you must update the `speechlmm/dataset/INPUTS_TEXT_LIST.json` and `speechlmm/dataset/OUTPUTS_TEXT_LIST.json` files accordingly.

**NOTE #2**: usually we keep `OUTPUTS_TEXT_LIST` as `None`, meaning that the LLM should simply provide the correct answer without any prefix.
