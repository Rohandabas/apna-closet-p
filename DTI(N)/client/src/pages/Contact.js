import React from "react";
import Layout from "./../components/Layout/Layout";
import { BiMailSend, BiPhoneCall, BiSupport } from "react-icons/bi";

const Contact = () => {
  return (
    <Layout title={"Contact us"}>
      <div className="contactuss container mt-1 pt-5">
        <div className="row">
          <div className="col-md-6">
            <img
              src="/images/cont.png"
              alt="Contact Us"
              className="img-fluid rounded"
              style={{ width: '100%' }} 
            />
          </div>
          <div className="col-md-6" style={{}}>           
           <div className="contact-info">
              <h2 style={{backgroundColor:'skyblue', width:'78vh'}} className="p-2 text-white text-center mb-4">CONTACT US</h2>
              <p className="text-justify">
                Have questions about our products or services? Don't hesitate to reach out to us.
              </p>
              <div className="contact-item mt-4">
                <BiMailSend className="contact-icon" />
                <p className="contact-text">Email: <a href="mailto:help@matrix.com">help@matrix.com</a></p>
              </div>
              <div className="contact-item mt-3">
                <BiPhoneCall className="contact-icon" />
                <p className="contact-text">Phone: <a href="tel:+918802612515">+91-8802612515</a></p>
              </div>
              <div className="contact-item mt-3">
                <BiSupport className="contact-icon" />
                <p className="contact-text">Support: 1800-0000-0000 (toll free)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Contact;
