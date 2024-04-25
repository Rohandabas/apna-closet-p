import React from 'react';
import Layout from '../../components/Layout/Layout';
import { Link } from 'react-router-dom';
import './Rental.css'; // Import custom CSS for styling

const RentalPlansPage = () => {
  return (
    <Layout>
      <div style={{ height: '50px' }}></div> {/* Space*/}
      <div className="container rental-plans mt-3">
        <h1 className="text-center mb-4" style={{fontSize: '2.5rem', fontWeight: 'bold'}}>Choose Your Rental Plan</h1>
        <div className="plans-container">
          <div className="plan">
            <h2>1 Month Plan</h2>
            <p>Description of the 1-month plan...</p>
            <p><strong>Offer : </strong>(10% discount)</p>
            <Link to="/checkout" className="btn btn-primaryx">Select Plan</Link>
          </div>
          <div className="plan">
            <h2>3 Month Plan</h2>
            <p>Description of the 3-month plan...</p>
            <p><strong>Offer : </strong>(15% discount)</p>
            <Link to="/checkout" className="btn btn-primaryx">Select Plan</Link>
          </div>
          <div className="plan">
            <h2>6 Month Plan</h2>
            <p>Description of the 6-month plan...</p>
            <p><strong>Offer : </strong>(25% discount) </p>
            <Link to="/checkout" className="btn btn-primaryx">Select Plan</Link>
          </div>
          <div className="plan">
            <h2>9 Month Plan</h2>
            <p>Description of the 9-month plan...</p>
            <p><strong>Offer : </strong>(30% discount)</p>
            <Link to="/checkout" className="btn btn-primaryx">Select Plan</Link>
          </div>
          <div className="plan">
            <h2>12 Month Plan</h2>
            <p>Description of the 12-month plan...</p>
            <p><strong>Offer : </strong>(40% discount)</p>
            <Link to="/checkout" className="btn btn-primaryx">Select Plan</Link>
          </div>
        </div>
        <div className="text-center mt-4">
          <Link to="/" className="btn btn-primaryx">Back to Homepage</Link>
        </div>
      </div>
    </Layout>
  );
};

export default RentalPlansPage;
