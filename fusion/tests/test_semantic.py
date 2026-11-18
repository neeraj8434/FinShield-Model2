"""Unit tests for OCR field extraction and semantic validation."""

import unittest
from fusion.ocr import extract_fields
from fusion.semantic import validate


class TestExtractFields(unittest.TestCase):
    def test_aadhaar_number(self):
        text = "UID: 1234 5678 9012\nName: Test User"
        f = extract_fields(text, "aadhaar")
        self.assertEqual(f["id_number"], "123456789012")

    def test_pan_number(self):
        text = "PAN ABCDE1234F\nName: Test User"
        f = extract_fields(text, "pan")
        self.assertEqual(f["id_number"], "ABCDE1234F")

    def test_dob(self):
        text = "Date of Birth: 15/08/1995\nName: Rajesh"
        f = extract_fields(text)
        self.assertEqual(f["dob"], "15/08/1995")

    def test_name(self):
        text = "Name: Shloka Pol\nDOB: 01/01/2000"
        f = extract_fields(text)
        self.assertEqual(f["name"], "Shloka Pol")

    def test_no_fields(self):
        f = extract_fields("random gibberish")
        self.assertEqual(f["id_number"], "")
        self.assertEqual(f["dob"], "")
        self.assertEqual(f["name"], "")


class TestSemanticValidate(unittest.TestCase):
    def test_all_pass(self):
        ocr = {"name": "Rajesh Kumar", "dob": "15/08/1995", "id_number": "123456789012"}
        r = validate(ocr, "Rajesh Kumar", "15/08/1995", "1234 5678 9012", "aadhaar")
        self.assertFalse(r.has_critical_fail)
        statuses = {c.field: c.status for c in r.checks}
        self.assertEqual(statuses["name"], "pass")
        self.assertEqual(statuses["dob"], "pass")
        self.assertEqual(statuses["id_number"], "pass")

    def test_name_mismatch(self):
        ocr = {"name": "Rajesh Kumar", "dob": "", "id_number": ""}
        r = validate(ocr, "Completely Different", doc_type="aadhaar")
        self.assertTrue(r.has_critical_fail)
        name_check = next(c for c in r.checks if c.field == "name")
        self.assertEqual(name_check.status, "fail")

    def test_dob_mismatch(self):
        ocr = {"name": "", "dob": "01/01/2000", "id_number": ""}
        r = validate(ocr, form_dob="15/08/1995")
        self.assertTrue(r.has_critical_fail)

    def test_pan_format_valid(self):
        ocr = {"name": "", "dob": "", "id_number": "ABCDE1234F"}
        r = validate(ocr, doc_type="pan")
        fmt = next(c for c in r.checks if c.field == "id_format")
        self.assertEqual(fmt.status, "pass")

    def test_pan_format_invalid(self):
        ocr = {"name": "", "dob": "", "id_number": "12345ABCDE"}
        r = validate(ocr, doc_type="pan")
        fmt = next(c for c in r.checks if c.field == "id_format")
        self.assertEqual(fmt.status, "warn")

    def test_missing_ocr_fields(self):
        ocr = {"name": "", "dob": "", "id_number": ""}
        r = validate(ocr, "Rajesh", "01/01/2000", "123456789012")
        self.assertFalse(r.has_critical_fail)  # unknown is not critical
        for c in r.checks:
            if c.field in ("name", "dob", "id_number"):
                self.assertEqual(c.status, "unknown")


if __name__ == "__main__":
    unittest.main()
