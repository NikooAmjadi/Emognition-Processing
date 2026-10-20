# چارچوب ماژولار آزمایش‌های یادگیری ماشین پایان‌نامه

این پروژه مستقل از نام یا ساختار یک دیتاست خاص طراحی شده است. تمام اطلاعات مربوط
به ستون‌ها، اهداف، خانواده‌های ویژگی و نحوه ساخت نمونه‌ها در `SCHEMA` داخل
`config.py` تعریف می‌شود.

## اجرا

ابتدا فقط `config.py` را ویرایش کنید و سپس اجرا کنید:

```powershell
python run.py
```

هیچ پارامتر خط فرمانی لازم نیست. هر اجرا در یک پوشه جدید زیر مسیر
`PATHS.output_dir` ذخیره می‌شود و نتایج قبلی overwrite نمی‌شوند.

## Progress and resume

With `EVALUATION.show_progress=True`, `tqdm.auto` displays one updating
fit-based progress bar. Its status contains the run percentage, active LOSO
fold, phase, and candidate. In Colab it uses the notebook-aware renderer. Model,
representation, mode, subject, fold, completion score, warnings, and errors are
written as durable timestamped events in `run.log`; the console keeps only
target and outer-fold milestones.

With `OUTPUT.save_checkpoints=True`, candidate-level checkpoints are committed
transactionally to a single `checkpoint.sqlite` database after inner-CV tuning
and again after outer refit/evaluation. If execution is interrupted, set only
this value in
`config.py` to the existing output directory and run `python run.py` again:

```python
PATHS = PathsConfig(
    # keep the same inputs/output settings as the original run
    resume_from=PROJECT_DIR / "model_results" / "20261008T181957Z_173b798590",
    # ...
)
```

The resume is refused if the scientific configuration or any input-file hash
differs from the original run. Set `resume_from=None` when starting a new run.
`OUTPUT.keep_checkpoint_after_completion=False` removes the database only after
a run is successfully finalized. Runs created before SQLite checkpoint support
cannot resume work retroactively.

## Nested model selection

### Primary classification metric

The primary classification metric is `subject_class_macro_recall`. Recall is
calculated for each subject/class combination that is present, averaged across
subjects within each class, and then averaged across classes. A subject is not
excluded merely because that subject does not contain every class. The same
metric is used for inner selection, final evaluation, and bootstrap confidence
intervals.

### Interim GCN/GAT input

For the current descriptor-level comparison, GCN and GAT receive the same
`hc`, `nvg`, `hvg`, `hc+nvg`, and `hc+hvg` columns as the classical models.
These columns are aligned into BVP, EDA, and TEMP sensor nodes; fusion blocks
are concatenated per sensor, structurally missing descriptor slots receive a
presence mask, and every selected input column is used exactly once. The three
sensor nodes use complete directed inter-sensor edges. This interim graph must
not be described as the original sample-level NVG/HVG topology.

هر سطر `CANDIDATES` یک pipeline ازپیش‌تعریف‌شده شامل input، input mode،
representation و model است. در هر Outer LOSO، grid پارامترهای هر candidate به‌طور
مستقل و فقط داخل Inner grouped CV ارزیابی می‌شود. سپس بهترین پارامترهای همان
candidate روی تمام Outer Train بازآموزی می‌شوند و آن candidate روی Outer Test
ارزیابی می‌شود. در نتیجه همه candidateها برای همه آزمودنی‌ها پیش‌بینی خارج از
نمونه دارند و می‌توان آن‌ها را به‌صورت paired مقایسه کرد.

معیار انتخاب در سطح trial و با وزن برابر subjectها محاسبه می‌شود؛ بنابراین
candidateهای دارای تعداد window متفاوت براساس تعداد ردیف بیشتر برتری مصنوعی
نمی‌گیرند. خروجی پیش‌فرض عمداً کوچک و بدون فایل‌های تکراری است:

فضای پیش‌فرض فعلی تمام گزینه‌های تعریف‌شده در `MODELS` را روی ورودی تنظیم‌شده
گسترش می‌دهد. تنظیم فعلی فقط پنجرهٔ ۶۰ ثانیه با هم‌پوشانی ۳۰ ثانیه را استفاده
می‌کند و ۷۰ candidate در هر target می‌سازد. پیش از اجرای
نهایی پایان‌نامه، این فهرست باید براساس فرضیه‌های ازپیش‌تعیین‌شده محدود و سپس
قفل شود. `quick=True` یا `max_outer_folds>0` فقط برای تست تشخیصی است و نتیجه آن
قابل گزارش به‌عنوان نتیجه نهایی نیست.

- `summary.csv`
- `subject_results.csv` (metrics and best parameters for every candidate/subject)
- `oof_trial_predictions.csv`
- `paired_statistical_tests.csv` (when statistical tests are enabled)
- `run_manifest.json`
- `run.log`
- `checkpoint.sqlite` while a run is active (removed after successful completion
  by the default `keep_checkpoint_after_completion=False` setting)

Window predictions, every inner-grid score, split assignments, and per-fold
SHAP/permutation tables are disabled by default. Enable their corresponding
`OUTPUT` switches only for a focused diagnostic run.

## تغییر دیتاست

برای یک CSV جدید این بخش‌های `config.py` را تنظیم کنید:

- `PATHS.input_files` یا `PATHS.input_globs`
- `SCHEMA.subject_column`
- `SCHEMA.trial_id_columns`
- `SCHEMA.target_columns`
- `SCHEMA.feature_prefixes`
- `SCHEMA.representations`
- `SCHEMA.expected_feature_counts`

اگر ساختار فایل با CSV عمومی قابل بیان نباشد، یک تابع adapter با امضای
`loader(path, config) -> Dataset` بسازید و مقدار `SCHEMA.adapter` را به فرم
`"my_package.my_adapter:load_dataset"` قرار دهید. برای این کار نیازی به تغییر
pipeline نیست. مدل‌ها و ارزیابی‌ها نباید نام دیتاست یا نام ستون‌های اختصاصی آن
را بدانند.

## نوع مسئله

مقدار `TASK.kind` یکی از این موارد است:

- `binary`
- `ternary`
- `multiclass`
- `regression`

تعریف هر چهار حالت از قبل در `thesis_ml/task_presets.py` قرار دارد؛ بنابراین در
استفادهٔ عادی فقط مقدار `TASK.kind` را تغییر دهید:

- `binary`: کم = ۱ تا ۴، زیاد = ۶ تا ۹، و مقدار ۵ حذف می‌شود.
- `ternary`: کم = ۱ تا ۳، متوسط = ۴ تا ۶، زیاد = ۷ تا ۹.
- `multiclass`: هر یک از مقادیر ۱ تا ۹ یک کلاس مستقل است.
- `regression`: مقدار عددی خام بدون گروه‌بندی یا حذف مقدار ۵ استفاده می‌شود.

نام مجموعهٔ پیش‌فرض‌ها `sam_1_9` است و برای دیتاستی با مقیاس هدف متفاوت می‌توان
یک مجموعهٔ جدید در همان فایل افزود. تعریف resolveشده در manifest هر اجرا ثبت
می‌شود. Softmax به‌صورت خودکار برای خروجی مدل‌های عصبی چندکلاسه اعمال می‌شود.

## افزودن مدل

مدل‌های موجود در `thesis_ml/models/` قرار دارند و از طریق
`thesis_ml/models/registry.py` ساخته می‌شوند. مدل جدید باید رابط استاندارد
`fit` و `predict` و در صورت classification، `predict_proba` را ارائه کند.
مدل‌های فعلی عبارت‌اند از SVM، Random Forest، KNN، XGBoost، MLP اسکیکت‌لرن،
MLP مبتنی بر PyTorch، GCN و GAT.

## تحلیل و خروجی

فعال یا غیرفعال‌کردن SHAP، permutation importance، correlation و آزمون‌های
آماری از `ANALYSES` انجام می‌شود؛ bootstrap در `EVALUATION` و فایل‌های خروجی
در `OUTPUT` کنترل می‌شوند.

`QUICK_RUN=True` automatically sets `bootstrap_repeats=0`. For the final thesis
run, changing `QUICK_RUN=False` restores 1,000 subject-cluster bootstrap
repetitions and therefore the confidence intervals; model fitting and selected
hyperparameters are otherwise unaffected by bootstrapping.

## تست سریع

```powershell
python -m unittest discover -s tests -v
```

تست integration از داده مصنوعی و SVM استفاده می‌کند و مدل‌های سنگین یا SHAP را
اجرا نمی‌کند.
