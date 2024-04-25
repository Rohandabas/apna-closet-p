import React from "react";
import Layout from "./../components/Layout/Layout";

const About = () => {
  return (
    <Layout title={"About us - Ecommer app"}>
      <div className="row contactus1 mt-5">
        <div className="col-md-6">
          <img
            src="/images/about.jpeg"
            alt="contactus"
            style={{ width: "100%" }}
          />
        </div>
        <div className="col-md-4" style={{ border: "3px solid rgb(2, 2, 57)", borderRadius: "10px", padding: "15px" }}>
      <p className="text-justify mt-2">
Welcome to Matrix – your premier destination for hassle-free tech rentals. Founded with a passion for innovation and a commitment to exceptional service, we offer a wide range of cutting-edge tech products available for rent, ensuring that our customers have access to the latest technology without the burden of ownership. 
      </p>
      <p className="text-justify mt-2">
      At Matrix, we pride ourselves on our dedication to customer satisfaction, providing personalized assistance and expert guidance every step of the way. With transparent pricing, flexible rental terms, and top-notch customer support, we strive to exceed your expectations at every opportunity. Join us on our mission to transform the way you experience technology. Contact us today to learn more or explore our range of tech rental options. Experience the future, one rental at a time, with Matrix.
      </p>
    </div>
      </div>
    </Layout>
  );
};

export default About;
