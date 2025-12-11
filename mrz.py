from imutils.contours import sort_contours
import pytesseract
import numpy as np
import imutils
import cv2
import os


class KernelCalculator:
	"""Responsible for computing morphological kernels based on image dimensions."""
	
	def __init__(self, kernel_scale=0.06, max_kernel=120, kernel_vert_arg=None):
		self.kernel_scale = kernel_scale
		self.max_kernel = max_kernel
		self.kernel_vert_arg = kernel_vert_arg
	
	def estimate_char_height(self, gray):
		"""Estimate character height from image to adapt vertical kernel."""
		H, W = gray.shape
		th_letters = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		smallK = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		th_letters = cv2.morphologyEx(th_letters, cv2.MORPH_OPEN, smallK, iterations=1)
		cnts_letters = cv2.findContours(th_letters.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts_letters = imutils.grab_contours(cnts_letters)
		heights = []
		for c2 in cnts_letters:
			(x2, y2, w2, h2) = cv2.boundingRect(c2)
			if w2 > 2 and h2 > 3 and h2 < (H * 0.5) and w2 < (W * 0.5):
				heights.append(h2)
		return int(np.median(heights)) if len(heights) > 0 else 6
	
	def compute_kernels(self, gray):
		"""Compute rectangular, square, and vertical kernels for morphological operations."""
		H, W = gray.shape
		
		# Compute rectangular kernel width
		rect_width = int(W * self.kernel_scale)
		rect_width = max(25, min(rect_width, self.max_kernel))
		
		# Compute vertical kernel height
		if self.kernel_vert_arg is not None:
			rect_vert = max(1, int(self.kernel_vert_arg))
		else:
			median_h = self.estimate_char_height(gray)
			if H > W:  # Portrait orientation
				rect_vert = max(3, min(int(median_h * 1.2), 25))
			else:  # Landscape orientation
				rect_vert = max(3, min(7, int(H * 0.01)))
		
		# Create kernels
		rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_width, rect_vert))
		sq_size = int(min(W, H) * 0.05)
		sq_size = max(21, min(sq_size, self.max_kernel))
		sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sq_size, sq_size))
		vertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rect_vert))
		
		return rectKernel, sqKernel, vertKernel


class MRZDetector:
	"""Responsible for detecting the MRZ region in passport images."""
	
	def __init__(self, gray, rectKernel, sqKernel, vertKernel, debug=False):
		self.gray = gray
		self.rectKernel = rectKernel
		self.sqKernel = sqKernel
		self.vertKernel = vertKernel
		self.H, self.W = gray.shape
		self.debug = debug
	
	def blackhat_gradient(self):
		"""Apply blackhat and Sobel gradient to detect text regions."""
		blackhat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, self.rectKernel)
		if self.debug:
			cv2.imshow("Blackhat", blackhat)
		
		grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
		grad = np.absolute(grad)
		(minVal, maxVal) = (np.min(grad), np.max(grad))
		grad = (grad - minVal) / (maxVal - minVal)
		grad = (grad * 255).astype("uint8")
		if self.debug:
			cv2.imshow("Gradient", grad)
		
		return grad
	
	def rect_close(self, grad):
		"""Apply rectangular closing and edge masking."""
		grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, self.rectKernel)
		thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		
		# Mask out edges: 5% from left and right
		edge_inset = int(max(self.W, self.H) * 0.04)
		thresh[:, :edge_inset] = 0
		thresh[:, self.W - edge_inset:] = 0
		if self.debug:
			cv2.imshow("Rect Close", thresh)

		return thresh
	
	def square_close(self, thresh):
		"""Apply square closing and edge masking."""
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.sqKernel)
		thresh = cv2.erode(thresh, None, iterations=2)
		if self.debug:
			cv2.imshow("Square Close", thresh)

		return thresh
	
	def vertical_open(self, thresh):
		"""Apply vertical opening to break vertical bridges."""
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.vertKernel)
		if self.debug:
			cv2.imshow("Vertical Open", thresh)
			
		return thresh
	
	def find_mrz_box(self, thresh):
		"""Find and return MRZ bounding box from thresholded image."""
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sort_contours(cnts, method="bottom-to-top")[0]
		
		mrzBox = None
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			percentWidth = w / float(self.W)
			percentHeight = h / float(self.H)
			if percentWidth > 0.8 and percentHeight > 0.04:
				mrzBox = (x, y, w, h)
				break
		
		return mrzBox
	
	def extract_mrz(self, image, mrzBox):
		"""Extract MRZ region from image using bounding box."""
		if mrzBox is None:
			return None
		
		(x, y, w, h) = mrzBox
		pX = int((x + w) * 0.03)
		pY = int((y + h) * 0.03)
		
		# Expand and clamp to image bounds
		x0 = max(0, x - pX)
		y0 = max(0, y - pY)
		x1 = min(self.W, x + w + pX)
		y1 = min(self.H, y + h + pY)
		
		new_w = x1 - x0
		new_h = y1 - y0
		if new_w <= 0 or new_h <= 0:
			return None
		
		return image[y0:y1, x0:x1]
	
	def detect(self, image):
		"""Main pipeline: detect and extract MRZ region."""
		# Preprocess grayscale image
		gray = cv2.GaussianBlur(self.gray, (3, 3), 0)
		self.gray = gray
		
		# Apply morphological operations
		grad = self.blackhat_gradient()
		thresh = self.rect_close(grad)
		thresh = self.square_close(thresh)
		thresh = self.vertical_open(thresh)
		
		# Find and extract MRZ
		mrzBox = self.find_mrz_box(thresh)
		if mrzBox is None:
			return None
		
		return self.extract_mrz(image, mrzBox)


class MRZPreprocessor:
	"""Responsible for preprocessing extracted MRZ images for OCR."""
	
	def __init__(self, scale_factor=4.8):
		self.scale_factor = scale_factor
	
	def to_grayscale(self, mrz):
		"""Convert MRZ image to grayscale."""
		return cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
	
	def upscale(self, mrz_gray):
		"""Upscale image using cubic interpolation."""
		h, w = mrz_gray.shape
		new_w = int(w * self.scale_factor)
		new_h = int(h * self.scale_factor)
		return cv2.resize(mrz_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
	
	def normalize(self, mrz_scaled):
		"""Normalize histogram (whitening)."""
		return cv2.normalize(mrz_scaled, None, 0, 255, cv2.NORM_MINMAX)
	
	def sharpen(self, mrz_normalized):
		"""Apply unsharp mask sharpening."""
		gaussian_blur = cv2.GaussianBlur(mrz_normalized, (0, 0), sigmaX=2.0)
		mrz_sharpened = cv2.addWeighted(mrz_normalized, 1.5, gaussian_blur, -0.5, 0)
		return np.clip(mrz_sharpened, 0, 255).astype(np.uint8)
	
	def preprocess(self, mrz):
		"""Full preprocessing pipeline."""
		mrz_gray = self.to_grayscale(mrz)
		mrz_scaled = self.upscale(mrz_gray)
		mrz_normalized = self.normalize(mrz_scaled)
		mrz_preprocessed = self.sharpen(mrz_normalized)
		return mrz_preprocessed


class MRZParser:
	"""Parse MRZ text for multiple document types (TD1, TD2, TD3, MRV-A, MRV-B)."""

	_weights = [7, 3, 1]

	def _char_value(self, ch: str) -> int:
		if ch.isdigit():
			return int(ch)
		if "A" <= ch <= "Z":
			return ord(ch) - ord("A") + 10
		return 0  # '<' or any other filler

	def _check_digit(self, field: str) -> str:
		total = 0
		for i, ch in enumerate(field):
			total += self._char_value(ch) * self._weights[i % 3]
		return str(total % 10)

	def _verify(self, field: str, digit: str) -> dict:
		expected = self._check_digit(field)
		found = digit or ""
		return {"expected": expected, "found": found, "valid": expected == found}

	def _clean(self, text: str) -> str:
		return text.replace("<<", " ").replace("<", " ").strip()

	def _split_names(self, names_raw: str):
		cleaned = self._clean(names_raw)
		if " " in cleaned:
			family, given = cleaned.split(" ", 1)
		else:
			family, given = cleaned, ""
		return family, given

	def detect_type(self, lines):
		line_count = len(lines)
		lengths = [len(l) for l in lines]

		if line_count == 3 and all(l >= 28 for l in lengths):
			return "TD1"
		if line_count == 2 and all(l >= 43 for l in lengths):
			return "MRV-A" if lines[0].startswith("V") else "TD3"
		if line_count == 2 and all(35 <= l < 43 for l in lengths):
			return "MRV-B" if lines[0].startswith("V") else "TD2"
		if line_count == 2 and all(33 <= l < 43 for l in lengths):
			return "MRV-B"
		return "UNKNOWN"

	def parse_td3(self, lines):
		row1, row2 = lines[-2:]
		names = row1[5:44]
		family, given = self._split_names(names)
		doc_field = row2[:9]
		doc_cd = row2[9:10]
		birth_field = row2[13:19]
		birth_cd = row2[19:20]
		exp_field = row2[21:27]
		exp_cd = row2[27:28]
		optional_field = row2[28:42]
		optional_cd = row2[42:43]
		final_cd = row2[43:44]
		composite = doc_field + doc_cd + birth_field + birth_cd + exp_field + exp_cd + optional_field + optional_cd
		return {
			"type": "TD3",
			"document_number": self._clean(doc_field),
			"issuing_country": self._clean(row1[2:5]),
			"nationality": self._clean(row2[10:13]),
			"birth_date": birth_field,
			"expiry_date": exp_field,
			"sex": row2[20:21].replace("<", ""),
			"optional_data": self._clean(optional_field),
			"optional_data_1": None,
			"optional_data_2": None,
			"family_name": family,
			"given_names": given,
			"hashes": {
				"document_number": self._verify(doc_field, doc_cd),
				"birth_date": self._verify(birth_field, birth_cd),
				"expiry_date": self._verify(exp_field, exp_cd),
				"optional_data": self._verify(optional_field, row2[42:43]),
				"final": self._verify(composite, final_cd),
			},
		}

	def parse_td2(self, lines):
		row1, row2 = lines[-2:]
		names = row1[5:36]
		family, given = self._split_names(names)
		doc_field = row2[:9]
		doc_cd = row2[9:10]
		birth_field = row2[13:19]
		birth_cd = row2[19:20]
		exp_field = row2[21:27]
		exp_cd = row2[27:28]
		optional_field = row2[28:35]
		final_cd = row2[35:36]
		composite = doc_field + doc_cd + birth_field + birth_cd + exp_field + exp_cd + optional_field
		return {
			"type": "TD2",
			"document_number": self._clean(doc_field),
			"issuing_country": self._clean(row1[2:5]),
			"nationality": self._clean(row2[10:13]),
			"birth_date": birth_field,
			"expiry_date": exp_field,
			"sex": row2[20:21].replace("<", ""),
			"optional_data": self._clean(optional_field),
			"optional_data_1": None,
			"optional_data_2": None,
			"family_name": family,
			"given_names": given,
			"hashes": {
				"document_number": self._verify(doc_field, doc_cd),
				"birth_date": self._verify(birth_field, birth_cd),
				"expiry_date": self._verify(exp_field, exp_cd),
				"optional_data": self._verify(optional_field, row2[34:35]),
				"final": self._verify(composite, final_cd),
			},
		}

	def parse_td1(self, lines):
		# TD1 has three lines; document number on line 1, names on line 3
		line1, line2, line3 = lines[-3:]
		names = line3[:30]
		family, given = self._split_names(names)
		doc_field = line1[5:14]
		doc_cd = line1[14:15]
		birth_field = line2[:6]
		birth_cd = line2[6:7]
		exp_field = line2[8:14]
		exp_cd = line2[14:15]
		optional1 = line1[15:30]
		optional2 = line2[18:29]
		final_cd = line2[29:30]
		composite = doc_field + doc_cd + birth_field + birth_cd + exp_field + exp_cd + optional1 + optional2
		return {
			"type": "TD1",
			"document_number": self._clean(doc_field),
			"issuing_country": self._clean(line1[2:5]),
			"nationality": self._clean(lines[-2][15:18]),
			"birth_date": birth_field,
			"expiry_date": exp_field,
			"sex": lines[-2][7:8].replace("<", ""),
			"optional_data": None,
			"optional_data_1": self._clean(optional1),
			"optional_data_2": self._clean(optional2),
			"family_name": family,
			"given_names": given,
			"hashes": {
				"document_number": self._verify(doc_field, doc_cd),
				"birth_date": self._verify(birth_field, birth_cd),
				"expiry_date": self._verify(exp_field, exp_cd),
				"optional_data": {"expected": None, "found": None, "valid": True},
				"final": self._verify(composite, final_cd),
			},
		}

	def parse_mrv_a(self, lines):
		row1, row2 = lines[-2:]
		names = row1[5:44]
		family, given = self._split_names(names)
		doc_field = row2[:9]
		doc_cd = row2[9:10]
		birth_field = row2[13:19]
		birth_cd = row2[19:20]
		exp_field = row2[21:27]
		exp_cd = row2[27:28]
		optional_field = row2[28:44]
		final_cd = row2[43:44]
		composite = doc_field + doc_cd + birth_field + birth_cd + exp_field + exp_cd + optional_field
		return {
			"type": "MRV-A",
			"document_number": self._clean(doc_field),
			"issuing_country": self._clean(row1[2:5]),
			"nationality": self._clean(row2[10:13]),
			"birth_date": birth_field,
			"expiry_date": exp_field,
			"sex": row2[20:21].replace("<", ""),
			"optional_data": self._clean(optional_field),
			"optional_data_1": None,
			"optional_data_2": None,
			"family_name": family,
			"given_names": given,
			"hashes": {
				"document_number": self._verify(doc_field, doc_cd),
				"birth_date": self._verify(birth_field, birth_cd),
				"expiry_date": self._verify(exp_field, exp_cd),
				"optional_data": self._verify(optional_field[:-1], optional_field[-1:]),
				"final": self._verify(composite, final_cd),
			},
		}

	def parse_mrv_b(self, lines):
		row1, row2 = lines[-2:]
		names = row1[5:36]
		family, given = self._split_names(names)
		doc_field = row2[:9]
		doc_cd = row2[9:10]
		birth_field = row2[13:19]
		birth_cd = row2[19:20]
		exp_field = row2[21:27]
		exp_cd = row2[27:28]
		optional_field = row2[28:36]
		final_cd = row2[35:36]
		composite = doc_field + doc_cd + birth_field + birth_cd + exp_field + exp_cd + optional_field
		return {
			"type": "MRV-B",
			"document_number": self._clean(doc_field),
			"issuing_country": self._clean(row1[2:5]),
			"nationality": self._clean(row2[10:13]),
			"birth_date": birth_field,
			"expiry_date": exp_field,
			"sex": row2[20:21].replace("<", ""),
			"optional_data": self._clean(optional_field),
			"optional_data_1": None,
			"optional_data_2": None,
			"family_name": family,
			"given_names": given,
			"hashes": {
				"document_number": self._verify(doc_field, doc_cd),
				"birth_date": self._verify(birth_field, birth_cd),
				"expiry_date": self._verify(exp_field, exp_cd),
				"optional_data": self._verify(optional_field[:-1], optional_field[-1:]),
				"final": self._verify(composite, final_cd),
			},
		}

	def parse(self, mrz_text: str):
		lines = [ln.strip() for ln in mrz_text.splitlines() if ln.strip()]
		if len(lines) < 2:
			return {
				"type": "UNKNOWN",
				"document_number": None,
				"issuing_country": None,
				"nationality": None,
				"birth_date": None,
				"expiry_date": None,
				"sex": None,
				"optional_data": None,
				"optional_data_1": None,
				"optional_data_2": None,
				"family_name": None,
				"given_names": None,
			}

		mrz_type = self.detect_type(lines)

		parsers = {
			"TD3": self.parse_td3,
			"TD2": self.parse_td2,
			"TD1": self.parse_td1,
			"MRV-A": self.parse_mrv_a,
			"MRV-B": self.parse_mrv_b,
		}

		parser_fn = parsers.get(mrz_type)
		if parser_fn is None:
			return {
				"type": "UNKNOWN",
				"document_number": None,
				"issuing_country": None,
				"nationality": None,
				"birth_date": None,
				"expiry_date": None,
				"sex": None,
				"optional_data": None,
				"optional_data_1": None,
				"optional_data_2": None,
				"family_name": None,
				"given_names": None,
			}

		parsed = parser_fn(lines)
		parsed["type"] = mrz_type
		return parsed


def mrz(file_data):
	"""
	Main function to process MRZ from uploaded file data.

	Args:
		file_data: Streamlit UploadedFile object or file-like object

	Returns:
		List containing [mrz_preprocessed, mrzText, parsed_data]
	"""
	if file_data is None:
		return [None, None, None]

	# Read image from uploaded file
	file_bytes = np.frombuffer(file_data.read(), np.uint8)
	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

	if image is None:
		print("[ERROR] could not decode image from uploaded file")
		return [None, None, None]

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Initialize kernel calculator and compute kernels
	kernel_calc = KernelCalculator(
		kernel_scale=0.06,
		max_kernel=120,
		kernel_vert_arg=None
	)
	rectKernel, sqKernel, vertKernel = kernel_calc.compute_kernels(gray)

	# Detect MRZ region
	detector = MRZDetector(gray, rectKernel, sqKernel, vertKernel, debug=False)
	mrz = detector.detect(image)

	if mrz is None:
		print("[INFO] MRZ could not be found")
		return [None, None, None]

	# Preprocess MRZ image
	preprocessor = MRZPreprocessor(scale_factor=4.8)
	mrz_preprocessed = preprocessor.preprocess(mrz)
	tessdata_dir = os.path.abspath("./custom/fast")
	tesseract_config = (
		f"--tessdata-dir {tessdata_dir} "
		"--psm 6 "
		"-l mrz "
		"-c load_system_dawg=F "
		"-c load_freq_dawg=F "
		"-c load_unambig_dawg=F "
		"-c load_punc_dawg=F "
		"-c load_number_dawg=F "
		"-c load_fixed_length_dawgs=F "
		"-c load_bigram_dawg=F "
		"-c wordrec_enable_assoc=F "
		"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
	)
	try:
		mrzText = pytesseract.image_to_string(mrz_preprocessed, config=tesseract_config)
	except Exception as e:
		print(f"[ERROR] pytesseract failed: {e}")
		return [None, None, None]

	mrzText = mrzText.replace(" ", "").strip()

	parser = MRZParser()
	parsed = parser.parse(mrzText)

	return [mrz_preprocessed, mrzText, parsed]
