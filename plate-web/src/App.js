import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Webcam from 'react-webcam';

const App = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [label, setLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [samples, setSamples] = useState([]);
  const webcamRef = React.useRef(null);

  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5050/api/samples");
      setSamples(res.data);
    } catch (err) {
      console.error("샘플 이미지 불러오기 실패:", err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    const res = await axios.post("http://127.0.0.1:5050/api/upload", formData);
    setLabel(res.data.label);
    setResultImage("http://127.0.0.1:5050" + res.data.result_path);
    setImagePreview(URL.createObjectURL(file));
    setLoading(false);
  };

  const handleCapture = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    const blob = await fetch(imageSrc).then(res => res.blob());
    const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    const res = await axios.post("http://127.0.0.1:5050/api/upload", formData);
    setLabel(res.data.label);
    setResultImage("http://127.0.0.1:5050" + res.data.result_path);
    setImagePreview(imageSrc);
    setLoading(false);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>차량 번호판 인식기 (OCR)</h2>

      <div>
        <input type="file" accept="image/*" onChange={handleFileUpload} />
        <button onClick={handleCapture}>웹캠 캡처</button>
      </div>

      <div style={{ marginTop: 20 }}>
        <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" width={320} />
      </div>

      {loading && <p>인식 중입니다... 잠시만 기다려주세요.</p>}

      {imagePreview && (
        <div style={{ marginTop: 30 }}>
          <h3>SAMPLE</h3>
          <div style={{ display: "flex", gap: 20 }}>
            <div>
              <p>원본 이미지</p>
              <img src={imagePreview} width={300} alt="original" />
            </div>
            <div>
              <p>번호판 인식 결과</p>
              <img src={resultImage} width={300} alt="result" />
              <p><strong>Label:</strong> {label}</p>
            </div>
          </div>
        </div>
      )}

      {/* ✅ 인식결과 샘플 섹션 */}
      <div style={{ marginTop: 50 }}>
        <h3>[인식결과 샘플]</h3>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {samples.map((sample, idx) => (
            <div key={idx}>
              <img src={`http://127.0.0.1:5050/api/image/${sample.result_url.split('/').pop()}`} width={200} />
              <p>{sample.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;

